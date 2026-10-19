"""Single-GPU experiment queue, with persistent stage status and fail-closed results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import read_json, write_json


def current_progress(directory):
    candidates = list(Path(directory).rglob('progress.json'))
    candidates = [p for p in candidates if 'source_snapshot' not in p.parts and 'invocation_sources' not in p.parts]
    if not candidates:
        return None
    try:
        return read_json(max(candidates, key=lambda p: p.stat().st_mtime))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def matching_process(needle):
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            command = (directory / 'cmdline').read_bytes().split(b'\0')
            if any(argument == needle.encode() or argument.endswith(b'/' + needle.encode()) for argument in command):
                return int(directory.name)
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=Path('/root/mia-paper-assets'))
    parser.add_argument('--wait-neighbour', action='store_true')
    parser.add_argument('--neighbour-directory', default='neighbour_agnews_fp32_full')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('The detached GPU queue requires Linux; run individual methods with reproductions.run')
    import fcntl
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, '-u', '-m', 'reproductions.queue', '--root', str(args.root), '--assets', str(args.assets)]
        if args.wait_neighbour:
            command.append('--wait-neighbour')
            command += ['--neighbour-directory', args.neighbour_directory]
        with (args.root / 'queue.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(args.root / 'queue_pid.json', {'pid': process.pid, 'launched_at': time.time()})
        print(f'Launched queue PID {process.pid}')
        return
    lock = (args.root / '.gpu_queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'started_at': time.time(), 'jobs': [], 'status': 'running'}

    def status(**updates):
        state.update(updates, updated_at=time.time())
        write_json(args.root / 'queue_status.json', state)

    if args.wait_neighbour:
        existing = args.root / 'runs' / args.neighbour_directory
        while not (existing / 'complete.json').exists():
            if (existing / 'failure.json').exists() or not matching_process('runs/' + args.neighbour_directory):
                state['jobs'].append({'method': 'neighbour', 'status': 'failed_or_stopped', 'directory': str(existing)})
                break
            status(current=args.neighbour_directory, progress=current_progress(existing))
            time.sleep(20)
        else:
            state['jobs'].append({'method': 'neighbour', 'status': 'completed', 'directory': str(existing)})
    configs = Path(__file__).parent / 'configs'
    jobs = [('spv', 'spv_agnews_gpt2xl.json', 'spv_agnews_gpt2xl_full', ['gpt2-xl']),
            ('ia', 'ia_nfcorpus_local.json', 'ia_nfcorpus_local_full', ['Qwen3-1.7B', 'flan-t5-large']),
            ('dcmi', 'dcmi_trec_covid_local.json', 'dcmi_trec_covid_local_full', ['Qwen3-1.7B', 'flan-t5-large'])]
    for method, filename, folder, needed in jobs:
        destination = args.root / 'runs' / folder
        job = {'method': method, 'directory': str(destination), 'status': 'waiting_for_assets'}
        state['jobs'].append(job)
        wait_started = time.time()
        while True:
            manifest_path = args.assets / 'asset_manifest.json'
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if all(name in manifest and any(f.endswith('.safetensors') for f in manifest[name]['files']) for name in needed):
                break
            if time.time() - wait_started > 6 * 3600:
                job.update(status='failed', reason='required public model assets not ready after six hours')
                break
            status(current=folder, progress={'stage': 'waiting_for_assets', 'required': needed,
                                             'elapsed_seconds': time.time() - wait_started})
            time.sleep(20)
        if job['status'] == 'failed':
            continue
        if method in {'ia', 'dcmi'}:
            preflight_marker = args.root / 'runs' / f'smoke_{method}' / 'complete.json'
            if not preflight_marker.exists():
                check_command = [sys.executable, '-u', '-m', 'reproductions.preflight',
                                 '--root', str(args.root), '--assets', str(args.assets), '--method', method]
                with (args.root / 'rag_preflight.log').open('a') as stream:
                    check = subprocess.run(check_command, stdout=stream, stderr=subprocess.STDOUT)
                if check.returncode != 0 or not preflight_marker.exists():
                    job.update(status='failed', reason='actual-model RAG preflight failed; full experiment not started')
                    continue
        if (destination / 'complete.json').exists():
            job['status'] = 'completed'
            continue
        command = [sys.executable, '-u', '-m', 'reproductions.run', '--config', str(configs / filename),
                   '--output', str(destination)]
        with (args.root / f'{folder}.log').open('a') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status='running', pid=process.pid, started_at=time.time())
            while process.poll() is None:
                status(current=folder, progress=current_progress(destination))
                time.sleep(20)
        job.update(status='completed' if process.returncode == 0 and (destination / 'complete.json').exists() else 'failed',
                   exit_code=process.returncode, finished_at=time.time())
        status(current=folder, progress=current_progress(destination))
    failed = [job for job in state['jobs'] if job['status'] != 'completed']
    status(status='completed' if not failed else 'finished_with_failures', current=None, progress=None)


if __name__ == '__main__':
    main()
