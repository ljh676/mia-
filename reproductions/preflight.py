"""Real-model RAG integration checks, explicitly separate from paper-size experiments."""
import argparse
from pathlib import Path
import time
from .common import read_json, write_json
from .run import run


def rag_preflight(root, assets, wait=False, requested_method='all'):
    root, assets = Path(root), Path(assets)
    started = time.time()
    while True:
        manifest_path = assets / 'asset_manifest.json'
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if all(name in manifest for name in ('Qwen3-1.7B', 'flan-t5-large')):
            break
        if not wait or time.time() - started > 6 * 3600:
            raise RuntimeError('RAG preflight needs verified local instruction and target models')
        write_json(root / 'rag_preflight_status.json', {'stage': 'waiting_for_assets', 'updated_at': time.time()})
        time.sleep(20)
    configs = Path(__file__).parent / 'configs'
    for method, filename in [('ia', 'ia_nfcorpus_local.json'), ('dcmi', 'dcmi_trec_covid_local.json')]:
        if requested_method not in {'all', method}:
            continue
        destination = root / 'runs' / f'smoke_{method}'
        if (destination / 'complete.json').exists():
            continue
        config = read_json(configs / filename)
        config.update(calibration_per_class=1, test_per_class=1,
                      corpus='/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/corpus.jsonl')
        config.pop('corpus_download_url', None)
        if method == 'ia':
            config['questions_per_document'] = 3
        else:
            config['theta_grid'] = [.04, .08]
            config['gamma_grid'] = [.1, .3, .5, .7, .9]
        write_json(root / 'rag_preflight_status.json', {'stage': f'testing_{method}', 'updated_at': time.time()})
        run(config, destination)
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    write_json(root / 'rag_preflight_status.json', {'stage': 'completed', 'updated_at': time.time(),
                'scope': 'four_documents_per_method_integration_only_not_paper_effectiveness'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=Path('/root/mia-paper-assets'))
    parser.add_argument('--wait-assets', action='store_true')
    parser.add_argument('--method', choices=['all', 'ia', 'dcmi'], default='all')
    args = parser.parse_args()
    rag_preflight(args.root, args.assets, args.wait_assets, args.method)


if __name__ == '__main__':
    main()
