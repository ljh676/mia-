"""Actual datastore membership, exact/near-duplicate checks and real retrieval."""
import json
import math
from pathlib import Path
import random
import re
import zipfile
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .common import text_hash, write_json, read_json, file_hash


def load_corpus(config, output):
    source = Path(config['corpus'])
    if not source.exists() and config.get('corpus_download_url'):
        import requests
        destination = Path(output) / 'download'
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / 'dataset.zip'
        if not archive.exists():
            temporary = archive.with_suffix('.part')
            with requests.get(config['corpus_download_url'], stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as zipped:
            names = [n for n in zipped.namelist() if n.endswith('/corpus.jsonl')]
            if len(names) != 1:
                raise ValueError('Expected exactly one BEIR corpus.jsonl')
            source.parent.mkdir(parents=True, exist_ok=True)
            with source.open('wb') as stream:
                with zipped.open(names[0]) as incoming:
                    for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                        stream.write(chunk)
        write_json(destination / 'source.json', {'url': config['corpus_download_url'],
                    'archive_sha256': file_hash(archive), 'corpus_sha256': file_hash(source)})
    if not source.exists():
        raise FileNotFoundError(source)
    corpus = {}
    hashes = set()
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            text = (str(row.get('title', '')) + '\n' + str(row['text'])).strip()
            key = text_hash(text)
            if key in hashes:
                continue
            hashes.add(key)
            corpus[str(row['_id'])] = {'text': text, 'sha256': key}
    return corpus


def prepare(config, output):
    output = Path(output)
    corpus = load_corpus(config, output)
    manifest_path = output / 'datastore.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['corpus_sha256'] != file_hash(config['corpus']):
            raise ValueError('Corpus changed since datastore creation')
        return corpus, manifest
    rng = random.Random(config['seed'])
    ids = sorted(corpus)
    rng.shuffle(ids)
    per_class = config['calibration_per_class'] + config['test_per_class']
    excluded_near = []
    if config['method'] == 'dcmi':
        boundary = int(len(ids) * .8)
        members, nonmembers = ids[:boundary], ids[boundary:]
    else:
        nonmembers = ids[:per_class]
        members = ids[per_class:]
        # Paper IA Section 6.1: exclude TF-IDF cosine near-duplicates of nonmembers.
        if config.get('near_duplicate_threshold'):
            matrix = TfidfVectorizer().fit_transform([corpus[i]['text'] for i in ids])
            nonmatrix = matrix[:per_class]
            keep = []
            for start in range(per_class, len(ids), 256):
                similarities = (matrix[start:start + 256] @ nonmatrix.T).max(axis=1).toarray().ravel()
                for identifier, similarity in zip(ids[start:start + 256], similarities):
                    (excluded_near if similarity >= config['near_duplicate_threshold'] else keep).append(identifier)
            members = keep
    if min(len(members), len(nonmembers)) < per_class:
        raise ValueError('Insufficient distinct datastore members/nonmembers; no silent downsizing')
    eval_mem = rng.sample(members, per_class)
    eval_non = rng.sample(nonmembers, per_class)
    records = []
    for label, selected in ((1, eval_mem), (0, eval_non)):
        for index, identifier in enumerate(selected):
            records.append({'id': identifier, 'label': label,
                            'split': 'calibration' if index < config['calibration_per_class'] else 'test'})
    member_set = set(members)
    assert all((record['id'] in member_set) == bool(record['label']) for record in records)
    manifest = {'corpus_sha256': file_hash(config['corpus']), 'deduplicated_corpus_size': len(corpus),
                'member_ids': members, 'nonmember_ids': nonmembers, 'excluded_near_duplicate_ids': excluded_near,
                'records': records, 'membership_source': 'actual_target_retrieval_index_document_ids',
                'datastore_ids_sha256': text_hash('\n'.join(sorted(members)))}
    write_json(manifest_path, manifest)
    return corpus, manifest


class Retriever:
    def __init__(self, config, corpus, member_ids, output):
        self.config, self.corpus, self.ids = config, corpus, member_ids
        self.kind = config['retriever']
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        texts = [corpus[i]['text'] for i in self.ids]
        if self.kind == 'bm25':
            self.vectorizer = CountVectorizer(dtype=np.float32)
            counts = self.vectorizer.fit_transform(texts)
            length = np.asarray(counts.sum(1)).ravel()
            average = length.mean()
            csc = counts.tocsc()
            df = np.diff(csc.indptr)
            self.idf = np.log(1 + (len(texts) - df + .5) / (df + .5))
            denominator = 1.2 * (1 - .75 + .75 * length / average)
            csr = counts.tocsr()
            document_rows = np.repeat(np.arange(csr.shape[0]), np.diff(csr.indptr))
            csr.data = (csr.data * 2.2 / (csr.data + denominator[document_rows]) * self.idf[csr.indices]).astype(np.float32)
            self.matrix = csr
        elif self.kind == 'dense':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config['retrieval_model'], device='cuda', local_files_only=True)
            cached = self.output / 'embeddings.npy'
            if cached.exists():
                self.matrix = np.load(cached)
            else:
                self.matrix = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                               show_progress_bar=True).astype(np.float32)
                np.save(cached, self.matrix)
            if len(self.matrix) != len(self.ids):
                raise ValueError('Cached dense index has wrong document count')
        else:
            raise ValueError(self.kind)
        write_json(self.output / 'index_manifest.json', {'kind': self.kind, 'ids': self.ids,
                   'bm25': {'k1': 1.2, 'b': .75, 'idf': 'lucene_log1p', 'tokenization': 'sklearn_word_unigram'} if self.kind == 'bm25' else None,
                   'dense_model': config.get('retrieval_model'), 'index_size': len(self.ids)})

    def search(self, query, k):
        if self.kind == 'bm25':
            query_counts = self.vectorizer.transform([query])
            query_counts.data[:] = 1
            values = (self.matrix @ query_counts.T).toarray().ravel()
        else:
            vector = self.model.encode([query], normalize_embeddings=True)[0]
            values = self.matrix @ vector
        order = np.argsort(-values, kind='stable')[:k]
        return [{'id': self.ids[i], 'score': float(values[i]), 'text': self.corpus[self.ids[i]]['text']} for i in order]
