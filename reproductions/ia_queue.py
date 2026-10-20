"""Check the repaired IA documents, then wait for the GPU queue before the full run."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import read_json, write_json, file_hash
from .queue import current_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    import fcntl
    root = args.root.resolve()
    os.chdir(Path(__file__).resolve().parents[1])
    own_lock = (root / '.ia_queue.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'status': 'running', 'started_at': time.time(), 'jobs': []}

    def status(**updates):
        state.update(updates, updated_at=time.time())
        write_json(root / 'ia_queue_status.json', state)

    def execute(config, output, job):
        if (output / 'complete.json').exists():
            return read_json(output / 'complete.json')
        config_path = root / 'repair_configs' / (output.name + '.json')
        write_json(config_path, config)
        with (root / (output.name + '.log')).open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'reproductions.run',
                        '--config', str(config_path), '--output', str(output)],
                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            job.update(pid=process.pid, directory=str(output))
            while process.poll() is None:
                status(current=output.name, progress=current_progress(output))
                time.sleep(5)
        job['exit_code'] = process.returncode
        marker = output / 'complete.json'
        return read_json(marker) if process.returncode == 0 and marker.exists() else None

    try:
        config = read_json(Path(__file__).parent / 'configs/ia_nfcorpus_local.json')
        smoke_config = dict(config, calibration_per_class=1, test_per_class=1)
        smoke = root / 'runs/smoke_ia_repair_v4'
        # Import generation evidence only. Failed-sample ledgers and RAG scores
        # stay in the old run, so those documents are actually retried here.
        old = root / 'runs/smoke_ia_repair_v2/generated_queries'
        imports = []
        if old.exists():
            for source in sorted(old.glob('*.json')):
                if source.name.endswith('.failed.json'):
                    continue
                target = smoke / 'generated_queries' / source.name
                copied = not target.exists()
                if copied:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                imports.append({'source': str(source), 'destination': str(target), 'sha256': file_hash(source),
                                'action': 'copied' if copied else 'retained_existing_cache',
                                'destination_sha256': file_hash(target)})
        write_json(smoke / 'generation_cache_imports.json', imports)
        # This short four-document check can share a 24 GB inference GPU when
        # enough memory is free. The formal experiment always takes the queue lock.
        while True:
            free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                         '--format=csv,noheader,nounits'], text=True)
            if int(free.splitlines()[0]) >= 11000:
                break
            status(status='waiting_for_memory', current=smoke.name)
            time.sleep(10)
        job = {'phase': 'four_document_check', 'status': 'running'}
        state['jobs'].append(job)
        status(status='running', current=smoke.name)
        result = execute(smoke_config, smoke, job)
        if not result or result['status'] != 'completed':
            job['status'] = 'failed'
            status(status='failed', reason='All four documents must have 30 validated questions and real RAG scores')
            return
        job['status'] = 'completed'
        gpu_lock = (root / '.gpu_queue.lock').open('a')
        while True:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status(status='waiting_for_gpu_queue', current='ia_formal', progress=None)
                time.sleep(10)
        destination = root / 'runs/ia_nfcorpus_local_full_repair_v4'
        for relative in ['data/datastore.json', 'index/embeddings.npy']:
            source = root / 'runs/ia_nfcorpus_local_full' / relative
            target = destination / relative
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        job = {'phase': 'formal', 'status': 'running'}
        state['jobs'].append(job)
        status(status='running', current=destination.name)
        result = execute(config, destination, job)
        job['status'] = result['status'] if result else 'failed'
        status(status=job['status'], current=None, progress=None, finished_at=time.time())
    except Exception as error:
        status(status='failed', error_type=type(error).__name__, message=str(error))
        raise


if __name__ == '__main__':
    main()
