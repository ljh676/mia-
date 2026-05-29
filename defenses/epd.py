
"""Enhanced EPD: retrieval-aware privacy-preserving generation defense.

EPD ??????????????????? RAG utility ??????
final answer ? retrieved context / member-specific evidence ????

?????
- target answer?????? RAG retrieval + Phi3 answer?? utility ???? leakage?
- base answer?????? retrieved docs???? query + Phi3 internal knowledge?
- judge answer??? retrieval overlap / confidence / specificity ????? target/base?
  ???? paraphrase??? exact phrase overlap?
"""

import glob
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Tuple

from .base import BaseDefense
from utils.metrics import compute_metrics
from utils.utility_metrics import response_length


class EPDDefense(BaseDefense):
    """RAG ?? inference-time ensemble privacy defense?"""

    defense_name = "epd"

    def prepare(self):
        """??????????"""
        self.real_rerun = bool(self.config.get("real_rerun", True))
        self.project_path = self.config.get("project_path", "/root/autodl-tmp/unified_mia_framework")
        self.rag_project_path = self.config.get("rag_project_path", "/root/autodl-tmp/RAG_MIA")
        self.output_path = self.config.get("output_path", "/root/autodl-tmp/unified_mia_framework/outputs/epd_result.json")
        self.retriever = self.config.get("retriever", "bge")
        self.retrieval_k = int(self.config.get("retrieval_k", 3))
        self.model_config_path = self.config.get("model_config_path", os.path.join(self.rag_project_path, "model_configs", "phi3_config.json"))
        self.extra_pythonpath = self.config.get("extra_pythonpath", "/root/autodl-tmp/phi3_compat_pkgs")
        self.max_records = int(self.config.get("max_records", 0))
        self.max_questions_per_doc = int(self.config.get("max_questions_per_doc", 0))
        self.max_examples = int(self.config.get("max_examples", 5))
        self.high_overlap_threshold = float(self.config.get("high_overlap_threshold", 0.18))
        self.base_preference_strength = float(self.config.get("base_preference_strength", 0.62))
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        return self

    def _latest_target_docs(self) -> str:
        explicit = self.config.get("target_docs_path")
        if explicit and os.path.exists(explicit):
            return explicit
        candidates = glob.glob(os.path.join(self.rag_project_path, "results", "target_docs", "*.json"))
        preferred = [p for p in candidates if "M100-N10" in os.path.basename(p)]
        if preferred:
            candidates = preferred
        if not candidates:
            return ""
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def _load_target_docs(self) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        path = self._latest_target_docs()
        if not path:
            return "", {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return path, data if isinstance(data, dict) else {str(i): v for i, v in enumerate(data) if isinstance(v, dict)}

    def _load_corpus(self) -> Dict[str, Dict[str, str]]:
        corpus_json = os.path.join(self.rag_project_path, "datasets", "nfcorpus", "corpus.json")
        corpus_jsonl = os.path.join(self.rag_project_path, "datasets", "nfcorpus", "corpus.jsonl")
        if os.path.exists(corpus_json):
            with open(corpus_json, "r", encoding="utf-8") as f:
                return json.load(f)
        corpus = {}
        with open(corpus_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                doc_id = item.get("_id") or item.get("id")
                corpus[str(doc_id)] = {"title": item.get("title", ""), "text": item.get("text", "")}
        return corpus

    def _ensure_rag_imports(self):
        if self.extra_pythonpath and os.path.isdir(self.extra_pythonpath) and self.extra_pythonpath not in sys.path:
            sys.path.insert(0, self.extra_pythonpath)
        if self.rag_project_path not in sys.path:
            sys.path.insert(0, self.rag_project_path)

    def _create_model(self):
        self._ensure_rag_imports()
        old_cwd = os.getcwd()
        os.chdir(self.rag_project_path)
        try:
            from src.models import create_model
            model = create_model(self.model_config_path)
            model.to(getattr(model, "device", None))
            return model
        finally:
            os.chdir(old_cwd)

    def _wrap_rag_prompt(self, query: str, contexts: List[str]) -> str:
        self._ensure_rag_imports()
        from src.prompts import wrap_prompt
        return wrap_prompt(query, contexts, prompt_id=4, context_free_response=False)

    def generate_target_answer(self, query: str, contexts: List[str], model=None) -> str:
        """target answer ?? RAG context?? high-utility / high-leakage ???"""
        if model is None:
            return "I don't know"
        return model.query(self._wrap_rag_prompt(query, contexts), max_output_tokens=5)

    def generate_base_answer(self, query: str, model) -> str:
        """base answer ????? retrieved docs / RAG context?"""
        prompt = (
            "Answer ONLY with one of: Yes, No, or I don't know.\n"
            "You must not use retrieved documents or any provided context.\n"
            "Answer from general knowledge only. If the question depends on a specific document, say I don't know.\n\n"
            f"Question:\n{query}\n\nAnswer:"
        )
        return model.query(prompt, max_output_tokens=5)

    def judge_answer(self, query: str, target_answer: str, base_answer: str, target_loss: float, base_loss: float, model, target_overlap: float = 0.0, base_overlap: float = 0.0) -> str:
        """judge prompt ???? privacy-preserving ??? overlap penalty?"""
        prompt = f"""You are a privacy-preserving judge.

You must:
- preserve factual correctness
- reduce membership leakage
- avoid exact copied phrases
- avoid highly specific retrieved wording
- prefer generalized or paraphrased answers
- reduce overlap with retrieved context

Question:
{query}

Answer A from target RAG model:
{target_answer}

Answer B from base model:
{base_answer}

Target answer loss:
{target_loss:.4f}

Base answer loss:
{base_loss:.4f}

Target retrieval overlap:
{target_overlap:.4f}

Base retrieval overlap:
{base_overlap:.4f}

If Answer A has high retrieval overlap or very high confidence, prefer Answer B or a generalized answer.
Answer ONLY with one of: Yes, No, or I don't know.

Final answer:"""
        return model.query(prompt, max_output_tokens=5)

    def _extract_yes_no(self, value: Any) -> str:
        text = str(value or "")
        if re.search(r"i\s*(do\s*)?n't\s+know|i\s+do\s+not\s+know|unknown", text, re.IGNORECASE):
            return "I don't know"
        match = re.search(r"\b(Yes|No)\b", text, re.IGNORECASE)
        return match.group(1).capitalize() if match else "I don't know"

    def _answer_loss_proxy(self, answer: str) -> float:
        normalized = self._extract_yes_no(answer)
        return 1.0 if normalized == "I don't know" else 0.2

    def _tokens(self, value: Any) -> List[str]:
        stop = {"the", "and", "for", "with", "that", "this", "from", "into", "does", "what", "which", "answer", "yes", "no", "know"}
        return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_\-']+", str(value or "")) if t.lower() not in stop]

    def _ngrams(self, tokens: List[str], n: int) -> set:
        return {tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens) - n + 1))}

    def _lcs_ratio(self, left: List[str], right: List[str]) -> float:
        if not left or not right:
            return 0.0
        # ??????? 128 ? token?
        left = left[:128]
        right = right[:128]
        prev = [0] * (len(right) + 1)
        for a in left:
            curr = [0]
            for j, b in enumerate(right, 1):
                curr.append(prev[j-1] + 1 if a == b else max(prev[j], curr[-1]))
            prev = curr
        return prev[-1] / max(1, min(len(left), len(right)))

    def compute_retrieval_overlap(self, answer: str, contexts: List[str]) -> float:
        """?? token/phrase/LCS/exact span reuse ? retrieval overlap score?"""
        answer_tokens = self._tokens(answer)
        context_text = " ".join(contexts)
        context_tokens = self._tokens(context_text)
        if not answer_tokens or not context_tokens:
            return 0.0
        token_overlap = len(set(answer_tokens) & set(context_tokens)) / max(1, len(set(answer_tokens)))
        phrase_scores = []
        for n in (2, 3):
            ans_ng = self._ngrams(answer_tokens, n)
            ctx_ng = self._ngrams(context_tokens, n)
            phrase_scores.append(len(ans_ng & ctx_ng) / max(1, len(ans_ng)))
        phrase_overlap = max(phrase_scores) if phrase_scores else 0.0
        lcs = self._lcs_ratio(answer_tokens, context_tokens)
        answer_lower = " ".join(answer_tokens)
        context_lower = " ".join(context_tokens)
        exact_span = 1.0 if len(answer_tokens) >= 4 and answer_lower in context_lower else 0.0
        return float(0.35 * token_overlap + 0.30 * phrase_overlap + 0.25 * lcs + 0.10 * exact_span)

    def _specificity(self, answer: str) -> float:
        """?????????????????????????"""
        text = str(answer or "")
        if not text:
            return 0.0
        tokens = re.findall(r"[A-Za-z0-9_\-']+", text)
        if not tokens:
            return 0.0
        long_terms = sum(1 for t in tokens if len(t) >= 11 or "-" in t)
        numbers = sum(1 for t in tokens if re.search(r"\d", t))
        caps = sum(1 for t in tokens if t[:1].isupper() and t.lower() not in {"yes", "no", "i"})
        return min(1.0, (long_terms + numbers + 0.5 * caps) / max(1, len(tokens)))

    def paraphrase_response(self, answer: str) -> str:
        """???? paraphrase??? Yes/No/IDK ??????????"""
        normalized = self._extract_yes_no(answer)
        # ?? RAG_MIA/Phi3 pipeline ??????????? paraphrase ???????
        if normalized == "Yes":
            return "Yes"
        if normalized == "No":
            return "No"
        return "I don't know"

    def _stable_gate(self, *parts: Any) -> float:
        """??? gate?????????"""
        text = "|".join(str(p) for p in parts)
        value = sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 10000
        return value / 10000.0

    def adaptive_fusion(self, query: str, target_answer: str, base_answer: str, contexts: List[str], is_hit: bool, target_overlap: float, base_overlap: float, judge_raw: str) -> Tuple[str, Dict[str, float]]:
        """?? overlap/confidence ????? target/base/generalized answer?"""
        target_norm = self._extract_yes_no(target_answer)
        base_norm = self._extract_yes_no(base_answer)
        judge_norm = self._extract_yes_no(judge_raw)
        target_conf = 1.0 if target_norm in {"Yes", "No"} else 0.0
        leak_risk = 0.45 * float(is_hit) + 0.30 * target_conf + 0.25 * target_overlap
        gate = self._stable_gate(query, target_answer, base_answer)
        # ?????? base/generalized ????? base ?????????? I don't know?
        if leak_risk >= 0.55 and gate < self.base_preference_strength:
            # ????? gate ????? base/generalized????? target utility?
            chosen = base_norm if base_norm != "I don't know" else "I don't know"
            source = "base_or_generalized"
        else:
            chosen = target_norm
            source = "target_preserved"
        chosen = self.paraphrase_response(chosen)
        final_overlap = self.compute_retrieval_overlap(chosen, contexts)
        return chosen, {
            "leak_risk": float(leak_risk),
            "target_confidence": float(target_conf),
            "target_overlap": float(target_overlap),
            "base_overlap": float(base_overlap),
            "final_overlap": float(final_overlap),
            "source": source,
        }

    def _doc_score(self, final_answers: List[str], overlap_infos: List[Dict[str, Any]]) -> float:
        """??? score???? final answer ????????? retrieval hit?"""
        if not final_answers:
            return 0.0
        yes_rate = sum(1 for a in final_answers if self._extract_yes_no(a) == "Yes") / len(final_answers)
        confidence = sum(1 for a in final_answers if self._extract_yes_no(a) in {"Yes", "No"}) / len(final_answers)
        final_overlap = sum(float(x.get("final_overlap", 0.0)) for x in overlap_infos) / max(1, len(overlap_infos))
        leak_risk = sum(float(x.get("leak_risk", 0.0)) for x in overlap_infos) / max(1, len(overlap_infos))
        specificity = sum(self._specificity(a) for a in final_answers) / len(final_answers)
        retrieval_signal = sum(float(x.get("leak_risk", 0.0)) for x in overlap_infos) / max(1, len(overlap_infos))
        # balanced enhanced EPD score????? retrieval signal????? final answer ???????
        return float(0.30 * retrieval_signal + 0.35 * yes_rate + 0.20 * final_overlap + 0.10 * specificity + 0.05 * confidence)

    def _metrics_from_scores(self, scores: List[float], labels: List[int]) -> Tuple[List[int], float, Dict[str, float]]:
        if not scores:
            return [], 0.0, {}
        threshold = sorted(scores)[len(scores) // 2]
        predictions = [1 if v >= threshold else 0 for v in scores]
        return predictions, float(threshold), compute_metrics(labels, predictions, scores)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.real_rerun:
            return {
                "defense_name": self.defense_name,
                "status": "success",
                "real_rerun": False,
                "scores": data.get("scores", []),
                "predictions": data.get("predictions", []),
                "labels": data.get("labels", []),
                "metrics": data.get("metrics", {}),
                "notes": "EPD configured with real_rerun=false; no response fusion was applied.",
            }

        target_docs_path, target_docs = self._load_target_docs()
        if not target_docs:
            raise RuntimeError("EPD real_rerun=true but no RAG_MIA target_docs were found")
        corpus = self._load_corpus()
        model = self._create_model()
        records = list(target_docs.items())
        if self.max_records > 0:
            records = records[: self.max_records]

        epd_docs: Dict[str, Dict[str, Any]] = {}
        scores: List[float] = []
        labels: List[int] = []
        all_final_answers: List[str] = []
        total_queries = retrieval_hits = qa_correct = 0
        examples: List[Dict[str, Any]] = []
        target_overlaps: List[float] = []
        final_overlaps: List[float] = []
        semantic_matches = 0
        specificity_values: List[float] = []

        for doc_index, (doc_id, record) in enumerate(records):
            questions = list(record.get("questions") or [])
            retrieved_lists = list(record.get("retrieved_doc_ids") or [])
            target_answers = list(record.get("llm_responses") or [])
            if self.max_questions_per_doc > 0:
                questions = questions[: self.max_questions_per_doc]
                retrieved_lists = retrieved_lists[: self.max_questions_per_doc]
                target_answers = target_answers[: self.max_questions_per_doc]

            base_answers, judge_answers, raw_judge_answers = [], [], []
            target_losses, base_losses, overlap_infos = [], [], []
            for qi, query in enumerate(questions):
                ids = retrieved_lists[qi] if qi < len(retrieved_lists) else []
                ids = ids if isinstance(ids, list) else [ids]
                contexts = [corpus.get(str(rid), {}).get("text", "")[:2048] for rid in ids]
                is_hit = str(doc_id) in [str(x) for x in ids]
                target_answer = target_answers[qi] if qi < len(target_answers) else self.generate_target_answer(query, contexts, model)
                base_answer = self.generate_base_answer(query, model)
                target_loss = self._answer_loss_proxy(target_answer)
                base_loss = self._answer_loss_proxy(base_answer)
                target_overlap = self.compute_retrieval_overlap(target_answer, contexts)
                base_overlap = self.compute_retrieval_overlap(base_answer, contexts)
                judge_raw = self.judge_answer(query, target_answer, base_answer, target_loss, base_loss, model, target_overlap, base_overlap)
                final_answer, info = self.adaptive_fusion(query, target_answer, base_answer, contexts, is_hit, target_overlap, base_overlap, judge_raw)

                base_answers.append(base_answer)
                raw_judge_answers.append(judge_raw)
                judge_answers.append(final_answer)
                target_losses.append(target_loss)
                base_losses.append(base_loss)
                overlap_infos.append(info)
                all_final_answers.append(final_answer)
                target_overlaps.append(target_overlap)
                final_overlaps.append(info["final_overlap"])
                specificity_values.append(self._specificity(final_answer))
                if self._extract_yes_no(final_answer) == self._extract_yes_no(target_answer):
                    semantic_matches += 1

                expected = "Yes" if is_hit else "No"
                retrieval_hits += 1 if is_hit else 0
                qa_correct += 1 if self._extract_yes_no(final_answer) == expected else 0
                total_queries += 1

                if len(examples) < self.max_examples:
                    examples.append({
                        "query": query,
                        "retrieved_doc_ids": [str(x) for x in ids],
                        "target_answer": target_answer,
                        "base_answer": base_answer,
                        "judge_raw_answer": judge_raw,
                        "judge_answer": final_answer,
                        "target_loss": target_loss,
                        "base_loss": base_loss,
                        "target_retrieval_overlap": target_overlap,
                        "base_retrieval_overlap": base_overlap,
                        "final_retrieval_overlap": info["final_overlap"],
                        "overlap_reduction": target_overlap - info["final_overlap"],
                        "fusion_source": info["source"],
                        "context_excerpt": "\n".join(contexts)[:800],
                    })

            score = self._doc_score(judge_answers, overlap_infos)
            label = 1 if str(record.get("mem", "yes")).lower() in {"yes", "1", "true", "member"} else 0
            scores.append(score)
            labels.append(label)
            updated = dict(record)
            updated.update({
                "epd_base_answers": base_answers,
                "epd_raw_judge_answers": raw_judge_answers,
                "epd_judge_answers": judge_answers,
                "epd_target_losses": target_losses,
                "epd_base_losses": base_losses,
                "epd_overlap_infos": overlap_infos,
                "epd_score": score,
            })
            epd_docs[str(doc_id)] = updated
            print(f"[EPD-enhanced] doc={doc_index+1}/{len(records)} id={doc_id} questions={len(questions)} score={score:.4f}")

        predictions, threshold, metrics = self._metrics_from_scores(scores, labels)
        retrieval_hit_rate = retrieval_hits / total_queries if total_queries else 0.0
        qa_accuracy = qa_correct / total_queries if total_queries else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0
        std_score = math.sqrt(sum((v - mean_score) ** 2 for v in scores) / len(scores)) if scores else 0.0
        resp_stats = response_length(all_final_answers)
        target_overlap_mean = sum(target_overlaps) / len(target_overlaps) if target_overlaps else 0.0
        final_overlap_mean = sum(final_overlaps) / len(final_overlaps) if final_overlaps else 0.0
        semantic_similarity = semantic_matches / total_queries if total_queries else 0.0
        response_specificity = sum(specificity_values) / len(specificity_values) if specificity_values else 0.0
        utility_override = {
            "attack_name": data.get("attack_name", "rag_mia"),
            "confidence": {"mean_score": mean_score, "std_score": std_score, "max_score": max(scores) if scores else 0.0, "min_score": min(scores) if scores else 0.0},
            "response_length": resp_stats,
            "retrieval": {"hit_rate": retrieval_hit_rate, "avg_retrieved_docs": float(self.retrieval_k), "num_queries": total_queries},
            "qa": {"qa_accuracy": qa_accuracy, "correct": qa_correct, "total": total_queries},
            "retrieval_hit_rate": retrieval_hit_rate,
            "qa_accuracy": qa_accuracy,
            "avg_response_length": resp_stats.get("avg_response_length", 0.0),
            "mean_score": mean_score,
            "utility_score": float(0.40 * retrieval_hit_rate + 0.40 * qa_accuracy + 0.20 * semantic_similarity),
            "target_overlap": target_overlap_mean,
            "final_overlap": final_overlap_mean,
            "overlap_reduction": target_overlap_mean - final_overlap_mean,
            "semantic_similarity": semantic_similarity,
            "response_specificity": response_specificity,
            "real_rerun": True,
            "defense_name": "epd_enhanced",
        }
        out_docs = os.path.join(self.project_path, "outputs", "epd_enhanced_realrerun_target_docs.json")
        with open(out_docs, "w", encoding="utf-8") as f:
            json.dump(epd_docs, f, ensure_ascii=False, indent=2)
        return {
            "defense_name": self.defense_name,
            "status": "success",
            "real_rerun": True,
            "enhanced": True,
            "scores": scores,
            "predictions": predictions,
            "labels": labels,
            "metrics": metrics,
            "threshold": threshold,
            "utility_override": utility_override,
            "examples": examples,
            "target_docs_path": target_docs_path,
            "epd_target_docs_path": out_docs,
            "num_records": len(epd_docs),
            "num_queries": total_queries,
            "target_model": self.config.get("target_model", "phi3"),
            "base_model": self.config.get("base_model", "phi3"),
            "judge_model": self.config.get("judge_model", "phi3"),
            "notes": "EPD-enhanced real_rerun=true: retrieval-aware adaptive fusion and paraphrased final answers.",
        }

    def evaluate(self, before_result, after_result):
        before_metrics = before_result.get("metrics", {}) or {}
        after_metrics = after_result.get("metrics", {}) or {}
        auc_before = float(before_metrics.get("auc", 0.0))
        auc_after = float(after_metrics.get("auc", 0.0))
        acc_before = float(before_metrics.get("accuracy", 0.0))
        acc_after = float(after_metrics.get("accuracy", 0.0))
        return {"auc_drop": auc_before - auc_after, "accuracy_drop": acc_before - acc_after, "defense_effectiveness": max(0.0, auc_before - auc_after)}
