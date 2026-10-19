import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import shutil


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(part)
    return sha.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def append_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_seed(seed, identifier):
    return (seed + int(hashlib.sha256(str(identifier).encode()).hexdigest()[:8], 16)) % (2 ** 31)


def model_fingerprint(path):
    path = Path(path)
    files = sorted([*path.glob('*.safetensors'), *path.glob('*config.json')])
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError(f'No safetensors checkpoint at {path}')
    return {'path': str(path), 'files': {p.name: file_hash(p) for p in files}}


def initialize_run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError(f'Refusing to mix different configurations in {output}')
    write_json(config_path, config)
    current_source = {p.relative_to(Path(__file__).parent).as_posix(): file_hash(p)
                      for p in Path(__file__).parent.rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts}
    source_version = digest(current_source)
    invocation_snapshot = output / 'invocation_sources' / source_version / 'reproductions'
    if not invocation_snapshot.exists():
        shutil.copytree(Path(__file__).parent, invocation_snapshot,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    append_record(output / 'invocations.jsonl', {'started_at': time.time(), 'source_version': source_version})
    source_snapshot = output / 'source_snapshot' / 'reproductions'
    if not source_snapshot.exists():
        shutil.copytree(Path(__file__).parent, source_snapshot,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source_hashes = {p.relative_to(source_snapshot).as_posix(): file_hash(p)
                     for p in source_snapshot.rglob('*') if p.is_file()}
    write_json(output / 'source_sha256.json', source_hashes)
    versions = {}
    for name in ('torch', 'transformers', 'datasets', 'numpy', 'scipy', 'scikit-learn', 'sentence-transformers',
                 'huggingface-hub', 'pyarrow', 'safetensors', 'sentencepiece'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(output / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
                                            'packages': versions, 'config_sha256': digest(config)})
    return output


def evaluate(records, output, method, fixed_threshold=None, threshold_source='independent_calibration_only'):
    from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support
    import numpy as np
    calibration = [r for r in records if r['split'] == 'calibration']
    test = [r for r in records if r['split'] == 'test']
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate sample identifiers')
    if {r['label'] for r in calibration} != {0, 1} or {r['label'] for r in test} != {0, 1}:
        raise ValueError('Both membership classes required in calibration and test')
    if any(not np.isfinite(r['score']) for r in records):
        raise ValueError('Non-finite membership score')
    values = sorted({r['score'] for r in calibration})
    thresholds = [float(np.nextafter(values[0], -np.inf)), *values]
    threshold = (max(thresholds, key=lambda t: (sum((r['score'] > t) == r['label'] for r in calibration), t))
                 if fixed_threshold is None else float(fixed_threshold))
    labels = [r['label'] for r in test]
    scores = [r['score'] for r in test]
    predictions = [int(score > threshold) for score in scores]
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    result = {'method': method, 'status': 'completed', 'evaluation_scope': 'configured_single_setting',
              'counts': {'calibration': len(calibration), 'test': len(test),
                         'test_members': sum(labels), 'test_nonmembers': len(labels) - sum(labels)},
              'auc': float(roc_auc_score(labels, scores)), 'accuracy': float(accuracy_score(labels, predictions)),
              'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
              'threshold': threshold, 'threshold_source': threshold_source,
              'score_orientation': 'larger_means_member_predeclared',
              'tpr_at_fpr': {str(rate): float(max(tpr[fpr <= rate], default=0)) for rate in (.01, .001)},
              'empirical_fpr_resolution': 1 / (len(labels) - sum(labels)),
              'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}, 'finished_at': time.time()}
    write_json(Path(output) / 'result.json', result)
    return result
