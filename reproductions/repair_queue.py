"""Run fresh, audited IA/DCMI repair experiments inside an existing tmux session."""
import argparse
import json
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
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.gpu_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(Path(__file__).resolve().parents[1])
    configs = Path(__file__).parent / 'configs'
    state = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'running',
             'queue_type': 'rag_generation_repair_v2', 'jobs': []}

    def save(**updates):
        state.update(updates, updated_at=time.time())
        write_json(root / 'repair_queue_status.json', state)
        write_json(root / 'queue_status.json', state)
        write_json(root / 'repair_handoff.json', {'session': os.environ.get('MIA_TMUX_SESSION', 'mia-repair'),
                   'pid': os.getpid(), 'status': state['status'], 'code': str(Path.cwd()),
                   'updated_at': time.time()})

    def execute(config, destination, job):
        if (destination / 'complete.json').exists():
            return read_json(destination / 'complete.json')
        config_path = root / 'repair_configs' / (destination.name + '.json')
        write_json(config_path, config)
        command = [sys.executable, '-u', '-m', 'reproductions.run', '--config', str(config_path), '--output', str(destination)]
        print('START ' + destination.name, flush=True)
        with (root / (destination.name + '.log')).open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            job.update(pid=process.pid)
            while process.poll() is None:
                save(current=destination.name, progress=current_progress(destination))
                time.sleep(10)
        job['last_exit_code'] = process.returncode
        marker = destination / 'complete.json'
        if process.returncode or not marker.exists():
            return None
        return read_json(marker)

    try:
        import nltk
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
        tagger = Path(str(nltk.data.find('taggers/averaged_perceptron_tagger_eng')))
        write_json(root / 'repair_lexical_resource.json', {'nltk_version': nltk.__version__,
                   'wordnet_version': wn.get_version(), 'source': str(wn.root),
                   'archive_sha256': file_hash(wn.root.zipfile.filename) if hasattr(wn.root, 'zipfile') else None,
                   'pos_tagger_files': {p.name: file_hash(p) for p in tagger.glob('*.json')},
                   'use': 'direct adjective/adverb antonyms selected in context by the local model'})
        # Validate both repaired paths before starting the longer formal settings.
        for method, filename in [('ia', 'ia_nfcorpus_local.json'), ('dcmi', 'dcmi_trec_covid_local.json')]:
            config = read_json(configs / filename)
            job = {'method': method, 'phase': 'preflight', 'status': 'running'}
            state['jobs'].append(job)
            smoke = dict(config, calibration_per_class=1, test_per_class=1,
                         corpus='/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/corpus.jsonl')
            smoke.pop('corpus_download_url', None)
            if method == 'dcmi':
                smoke.update(theta_grid=[.04, .08], gamma_grid=[.1, .3, .5, .7, .9])
            destination = root / 'runs' / ('smoke_' + method + '_repair_v2')
            result = execute(smoke, destination, job)
            job.update(status='completed' if result and result['status'] == 'completed' else 'failed',
                       directory=str(destination), finished_at=time.time())
            save()
        for method, filename, original in [('dcmi', 'dcmi_trec_covid_local.json', 'dcmi_trec_covid_local_full'),
                                           ('ia', 'ia_nfcorpus_local.json', 'ia_nfcorpus_local_full')]:
            preflight = next(j for j in state['jobs'] if j['method'] == method and j['phase'] == 'preflight')
            job = {'method': method, 'phase': 'formal', 'status': 'running'}
            state['jobs'].append(job)
            if preflight['status'] != 'completed':
                job.update(status='failed', reason='strict four-document actual-model check did not pass')
                save()
                continue
            config = read_json(configs / filename)
            destination = root / 'runs' / (original + '_repair_v2')
            # Reuse only membership and dense vectors. No old generated questions,
            # edits or query signals enter the repaired experiment.
            original_directory = root / 'runs' / original
            for relative in ['data/datastore.json', 'index/embeddings.npy']:
                source, target = original_directory / relative, destination / relative
                if source.exists() and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            job['directory'] = str(destination)
            result = execute(config, destination, job)
            job.update(status=result['status'] if result else 'failed', finished_at=time.time())
            save()
        statuses = [j['status'] for j in state['jobs']]
        final = ('finished_with_failures' if 'failed' in statuses else
                 'finished_with_exclusions' if 'completed_with_exclusions' in statuses else 'completed')
        save(status=final, current=None, progress=None, finished_at=time.time())
        print('QUEUE FINISHED: ' + final, flush=True)
    except Exception as error:
        save(status='failed', error_type=type(error).__name__, message=str(error))
        raise


if __name__ == '__main__':
    main()
