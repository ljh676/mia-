
import json, random, os
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from beir.datasets.data_loader import GenericDataLoader

ROOT = Path(__file__).resolve().parents[1]
CORPUS_JSONL = ROOT / 'indexes' / 'nfcorpus_corpus.jsonl'
INDEX_PATH = ROOT / 'indexes' / 'nfcorpus_gte_small' / 'gte_Flat.index'
MODEL_PATH = ROOT / 'models' / 'gte-small'
DATA_PATH = ROOT / 'data' / 'nfcorpus_unzip' / 'nfcorpus'
OUT_DIR = ROOT / 'outputs'
OUT_DIR.mkdir(exist_ok=True)
SEED = 2024
SAMPLE_N = int(os.environ.get('DCMI_SAMPLE_N', '100'))
TOPK = 5
random.seed(SEED)
np.random.seed(SEED)

def perturb_query(text: str) -> str:
    words = text.split()
    if not words:
        return text
    k = max(1, int(0.03 * len(words)))
    positions = np.linspace(0, len(words) - 1, k, dtype=int).tolist()
    out = words[:]
    for pos in positions:
        out[pos] = 'not_' + out[pos]
    return ' '.join(out)

def load_jsonl(path):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def dot_score(q_emb, doc_vec):
    return float(np.dot(q_emb.astype('float32'), doc_vec.astype('float32')))

def optimize_threshold(labels, scores):
    lo, hi = min(scores), max(scores)
    thresholds = np.linspace(lo, hi, 401)
    best_thr, best_acc = thresholds[0], -1
    for thr in thresholds:
        preds = [1 if s > thr else 0 for s in scores]
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_thr, best_acc = float(thr), float(acc)
    return best_thr, best_acc

corpus_rows = load_jsonl(CORPUS_JSONL)
id_to_idx = {row['id']: i for i, row in enumerate(corpus_rows)}
all_doc_ids = list(id_to_idx)
corpus, queries, qrels = GenericDataLoader(data_folder=str(DATA_PATH)).load(split='test')
index = faiss.read_index(str(INDEX_PATH))
model = SentenceTransformer(str(MODEL_PATH), trust_remote_code=True, model_kwargs={'torch_dtype': 'float16'}).to('cuda')

eligible = [(qid, q) for qid, q in queries.items() if any(docid in id_to_idx and rel > 0 for docid, rel in qrels.get(qid, {}).items())]
eligible = eligible[:SAMPLE_N]
query_texts = [q for _, q in eligible]
pert_texts = [perturb_query(q) for q in query_texts]
q_emb = model.encode(query_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype('float32')
p_emb = model.encode(pert_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype('float32')

all_qids = list(queries)
all_qtexts = [queries[qid] for qid in all_qids]
all_emb = model.encode(all_qtexts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype('float32')
D, I = index.search(all_emb, TOPK)
hits = 0
for qid, idxs in zip(all_qids, I):
    rel = {docid for docid, score in qrels.get(qid, {}).items() if score > 0}
    got = {corpus_rows[int(i)]['id'] for i in idxs if int(i) >= 0}
    hits += bool(rel & got)
recall_at_5 = hits / len(all_qids)

mem, perturb_mem, nom, perturb_nom = [], [], [], []
examples = []
for row_i, ((qid, query), qe, pe) in enumerate(zip(eligible, q_emb, p_emb)):
    positives = [docid for docid, rel in qrels[qid].items() if rel > 0 and docid in id_to_idx]
    member_doc = positives[0]
    relset = set(positives)
    non_doc = random.choice([docid for docid in all_doc_ids if docid not in relset])
    for bucket, docid in [('member', member_doc), ('nonmember', non_doc)]:
        vec = np.empty(index.d, dtype='float32')
        index.reconstruct(id_to_idx[docid], vec)
        s0 = dot_score(qe, vec)
        s1 = dot_score(pe, vec)
        if bucket == 'member':
            mem.append(s0); perturb_mem.append(s1)
        else:
            nom.append(s0); perturb_nom.append(s1)
    if len(examples) < 3:
        examples.append({'qid': qid, 'query': query, 'perturbed': pert_texts[row_i], 'member_doc': member_doc, 'nonmember_doc': non_doc})

diff_member = (np.array(mem) - np.array(perturb_mem)).tolist()
diff_nonmember = (np.array(nom) - np.array(perturb_nom)).tolist()
labels = [1] * len(diff_member) + [0] * len(diff_nonmember)
scores = diff_member + diff_nonmember
thr, acc = optimize_threshold(labels, scores)
preds = [1 if s > thr else 0 for s in scores]
metrics = {
    'sample_n_per_class': len(diff_member),
    'threshold': thr,
    'accuracy': accuracy_score(labels, preds),
    'auc': roc_auc_score(labels, scores),
    'precision': precision_score(labels, preds, zero_division=0),
    'recall': recall_score(labels, preds, zero_division=0),
    'f1': f1_score(labels, preds, zero_division=0),
    'retrieval_recall_at_5': recall_at_5,
    'member_diff_mean': float(np.mean(diff_member)),
    'nonmember_diff_mean': float(np.mean(diff_nonmember)),
    'examples': examples,
}
out_json = OUT_DIR / 'min_dcmi_nfcorpus_metrics.json'
out_scores = OUT_DIR / 'min_dcmi_nfcorpus_scores.json'
out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
out_scores.write_text(json.dumps({'mem': mem, 'perturb_mem': perturb_mem, 'nom': nom, 'perturb_nom': perturb_nom, 'diff_member': diff_member, 'diff_nonmember': diff_nonmember}, indent=2), encoding='utf-8')
print('Optimized Threshold: %.6f' % metrics['threshold'])
print('Best Accuracy: %.2f%%' % (metrics['accuracy'] * 100))
print('AUC: %.4f' % metrics['auc'])
print('Precision: %.4f' % metrics['precision'])
print('Recall:    %.4f' % metrics['recall'])
print('F1-score:  %.4f' % metrics['f1'])
print('Retrieval recall@5: %.4f' % metrics['retrieval_recall_at_5'])
print('member diff mean: %.6f' % metrics['member_diff_mean'])
print('nonmember diff mean: %.6f' % metrics['nonmember_diff_mean'])
print('metrics_json:', out_json)
print('scores_json:', out_scores)
