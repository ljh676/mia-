"""Fetch public, revision-pinned assets without changing existing model directories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    parser.add_argument('--models', nargs='*', default=[])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    os.environ['HF_ENDPOINT'] = args.endpoint
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(endpoint=args.endpoint)
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / 'asset_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    jobs = [('fancyzhx/ag_news', 'dataset', 'ag_news')]
    jobs += [(name, 'model', name.split('/')[-1]) for name in args.models]
    manifest_lock = threading.Lock()

    def fetch(job):
        name, kind, folder = job
        previous = manifest.get(folder)
        if previous and previous['repo'] == name:
            revision = previous['revision']
        else:
            revision = api.repo_info(name, repo_type=kind).sha
        destination = args.root / folder
        print(f'Fetching {kind} {name}@{revision}', flush=True)
        patterns = (['data/*.parquet', 'README.md'] if kind == 'dataset' else
                    ['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'vocab.txt',
                     'spiece.model', 'tokenizer.model', 'LICENSE*', 'README.md'])
        snapshot_download(name, repo_type=kind, revision=revision, local_dir=destination,
                          allow_patterns=patterns, max_workers=3, endpoint=args.endpoint)
        files = {}
        for path in destination.rglob('*'):
            if path.is_file() and '.cache' not in path.parts:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(chunk)
                files[path.relative_to(destination).as_posix()] = {
                    'size': path.stat().st_size, 'sha256': digest.hexdigest()}
        with manifest_lock:
            manifest[folder] = {'repo': name, 'kind': kind, 'revision': revision,
                                'path': str(destination), 'files': files, 'verified_at': time.time()}
            temporary = manifest_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(manifest, indent=2) + '\n')
            temporary.replace(manifest_path)
        print(f'Verified {folder}: {len(files)} files', flush=True)
    def fetch_with_retry(job):
        for attempt in range(6):
            try:
                return fetch(job)
            except Exception as error:
                print(f'Asset transfer interrupted for {job[0]}: {type(error).__name__}; attempt {attempt + 1}/6', flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(30, 5 * (attempt + 1)))
    fetch_with_retry(jobs[0])
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(fetch_with_retry, jobs[1:]))


if __name__ == '__main__':
    main()
