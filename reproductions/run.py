"""Run complete method stages in a new, configuration-locked experiment directory."""
import argparse
from pathlib import Path
import time
import traceback
from .common import read_json, write_json, initialize_run


def run(config, output):
    output = initialize_run(config, output)
    try:
        if config['method'] in {'neighbour', 'spv'}:
            from .data import prepare
            from .training import train, self_prompt
            prepare(config, output / 'data')
            target = train(config, output / 'data', output / 'target')
            if config['method'] == 'neighbour':
                from .neighbour import attack
                result = attack(config, output / 'data', target, output / 'attack')
            else:
                from .spv import attack
                generated = self_prompt(config, target, output / 'data', output / 'self_prompt')
                reference = train(config, generated, output / 'reference', role='reference')
                result = attack(config, output / 'data', target, reference, output / 'attack')
        elif config['method'] in {'ia', 'dcmi'}:
            from .rag import run_rag
            result = run_rag(config, output)
        else:
            raise ValueError(config['method'])
        write_json(output / 'complete.json', {'status': result.get('status', 'completed'), 'finished_at': time.time(), 'result': result,
                                            'scope': 'one_configured_setting_not_all_paper_tables'})
    except Exception as error:
        write_json(output / 'failure.json', {'status': 'failed', 'type': type(error).__name__,
                                           'message': str(error), 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(read_json(args.config), args.output)


if __name__ == '__main__':
    main()
