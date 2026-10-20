"""Read persistent queue/training state; never infer success from stale score files."""
import argparse
import json
from pathlib import Path
import time
from .common import read_json, write_json
from .queue import current_progress


def collect(root):
    root = Path(root)
    report = {'observed_at': time.time(), 'queue': None, 'experiments': []}
    if (root / 'queue_status.json').exists():
        report['queue'] = read_json(root / 'queue_status.json')
    for directory in sorted((root / 'runs').glob('*')):
        if not directory.is_dir():
            continue
        record = {'name': directory.name, 'scope': 'integration_only' if directory.name.startswith('smoke_') else 'formal_single_setting',
                  'progress': current_progress(directory)}
        if (directory / 'superseded.json').exists():
            record.update(status='superseded', reason=read_json(directory / 'superseded.json')['reason'])
        elif (directory / 'complete.json').exists():
            complete = read_json(directory / 'complete.json')
            record.update(status=complete['status'], result=complete['result'])
        elif (directory / 'failure.json').exists():
            failure = read_json(directory / 'failure.json')
            record.update(status='failed_or_resuming', failure_type=failure['type'], message=failure['message'])
        else:
            record['status'] = 'in_progress_or_stopped_check_queue'
        report['experiments'].append(record)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = collect(args.root)
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
