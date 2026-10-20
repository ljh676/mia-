"""Bounded, resumable local generation with explicit failure and lexical evidence."""
import json
from pathlib import Path
import re
from .common import read_json, write_json, text_hash
from .generation import json_object


class UnscorableSample(RuntimeError):
    """A document could not satisfy the configured attack's input requirements."""


def valid_binary_question(question):
    return (isinstance(question, str) and len(question.strip()) >= 15
            and bool(re.match(r'^(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b', question.strip(), re.I))
            and not re.search(r'\b(?:passage|document|context)\b', question, re.I)
            and question.strip().endswith('?'))


def generate_ia_queries(generator, document, count, output, max_attempts=72):
    output = Path(output)
    legacy = None
    if output.exists():
        saved = read_json(output)
        if saved['document_sha256'] != text_hash(document) or len(saved['queries']) != count:
            raise ValueError('IA query cache identity mismatch')
        if all(valid_binary_question(q['probe']) and q['reference_answer'] in {'yes', 'no'} for q in saved['queries']):
            return saved
        legacy = saved
        # Preserve the evidence for rejected cached questions instead of silently
        # treating old "does the passage mention..." questions as valid inputs.
        write_json(output.with_suffix('.rejected_cache.json'), saved)
    partial = output.with_suffix('.v4.partial.json')
    old_partial = output.with_suffix('.v3.partial.json')
    state = (read_json(partial) if partial.exists() else legacy if legacy is not None else
             read_json(old_partial) if old_partial.exists() else
             {'document_sha256': text_hash(document), 'summary': None, 'queries': [],
              'generation_evidence': [], 'excluded_unknown_reference': [], 'attempts': 0})
    if state['document_sha256'] != text_hash(document):
        raise ValueError('IA partial cache identity mismatch')
    rejected = [q for q in state['queries'] if not valid_binary_question(q['probe'])
                or q['reference_answer'] not in {'yes', 'no'}]
    state['queries'] = [q for q in state['queries'] if q not in rejected]
    state.setdefault('rejected_cached_questions', []).extend(rejected)
    if state.get('generation_policy_version') != 5:
        state['previous_policy_attempts'] = state['attempts']
        state['attempts'] = 0
        state['generation_policy_version'] = 5
    seen = {text_hash(item['probe']) for item in state['queries']}
    chunks = [s.strip() for s in re.split(r'(?<=[.!?])\s+', document) if len(s.strip()) > 25]
    factual = [s for s in chunks if re.search(r'\b(?:is|are|was|were|has|have|had|showed|found|compared|occurred|emerged)\b', s, re.I)
               and not re.search(r'\b(?:this article|this review|are discussed|reviews available)\b', s, re.I)]
    chunks = factual or chunks
    # An old exhausted partial is resumable under the newer bounded budget.
    # Rotate only facts actually present in the passage: a mandatory "study
    # population" focus invented unanswerable questions for review abstracts.
    for attempt in range(state['attempts'], max_attempts):
        required = min(4, count - len(state['queries']))
        if required <= 0:
            break
        focus = chunks[attempt % len(chunks)] if chunks else document
        contrast = attempt % 3
        task = (
            'Ask about separate explicit facts and relationships in this passage.' if contrast == 0 else
            'Ask counterfactual yes/no questions: change ONE entity, relationship, direction or quantity '
            'to an alternative that this passage explicitly contradicts. The correct answers should be No.' if contrast == 1 else
            'Ask about specific comparisons or combinations of explicitly stated facts. Include both '
            'true claims and explicitly contradicted alternatives, without inventing unstated study details.')
        schema = ('{"questions":["..."]}' if state['summary'] else
                  '{"summary":"My inquiry concerns ...", "questions":["..."]}')
        prompt = (
            f'Write {required + 2} distinct YES/NO questions about the focus passage below. '
            'Every question must start with Is, Are, Was, Were, Did, Does, Has or Can and end with ?. '
            'Use precise subject names; each question must stand alone. Never write What, Why or How questions. '
            'Vary individual quantities, populations, time periods '
            'and comparisons explicitly given in the passage. Do not use outside knowledge or ask about database membership. '
            + task + ' '
            'Do not ask whether the document or passage mentions something. Do not invent a study population, '
            'control group, dose or result absent from the passage. '
            'Return JSON only: ' + schema + '. '
            'The summary should be a short retrieval description of the document topic.\n'
            'Document title/topic: ' + document.splitlines()[0] + '\n'
            + '\nFocus passage (use facts from THIS passage, not other parts of the document):\n' + focus
            + f'\nReturn {required + 2} NEW yes/no questions now. JSON only.')
        raw = generator.generate(prompt, 1200, temperature=.65 if attempt else 0.)
        state['generation_evidence'].append({'prompt': prompt, 'raw': raw, 'attempt': attempt,
                                              'strategy': 'factual_sentences_no_meta_questions_v5',
                                              'temperature': .65 if attempt else 0.})
        try:
            candidates = json_object(raw)
        except (ValueError, TypeError):
            candidates = {}
        if not isinstance(candidates, dict):
            candidates = {}
        if not state['summary'] and isinstance(candidates.get('summary'), str):
            state['summary'] = candidates['summary'].strip()
        if not state['summary']:
            state['attempts'] = attempt + 1
            write_json(partial, state)
            continue
        proposed = candidates.get('questions', [])
        for probe in proposed if isinstance(proposed, list) else []:
            if not valid_binary_question(probe) or text_hash(probe) in seen:
                continue
            probe = probe.strip()
            seen.add(text_hash(probe))
            query = state['summary'] + ' ' + probe + " Please answer with Yes, No, or I don't know."
            gold, evidence = generator.answer(document, probe)
            if gold not in {'yes', 'no'}:
                state['excluded_unknown_reference'].append({'probe': probe, 'reference': evidence})
                continue
            state['queries'].append({'probe': probe, 'query': query,
                                     'reference_answer': gold, 'reference_evidence': evidence})
            if len(state['queries']) == count:
                break
        state['attempts'] = attempt + 1
        write_json(partial, state)
    if len(state['queries']) != count:
        write_json(output.with_suffix('.failed.json'), state)
        raise UnscorableSample(f'Local model produced only {len(state["queries"])}/{count} validated IA questions')
    state['construction'] = 'factual_sentences_no_meta_questions_v5_full_document_reference'
    write_json(output, state)
    return state


def lexical_candidates(document):
    """Direct WordNet adjective/adverb antonyms, with source sense and exact positions.

    Lexical validity does not certify contextual word sense; the local model must
    separately select suitable edits. Numbers, acronyms and names are excluded.
    """
    from nltk.corpus import wordnet as wn
    import nltk
    wn.ensure_loaded()
    candidates = []
    spans = list(re.finditer(r"\b\w+(?:[-']\w+)*\b", document))
    tagged = nltk.pos_tag([match.group() for match in spans])
    for position, (match, (_, tag)) in enumerate(zip(spans, tagged)):
        word = match.group()
        if word.isupper() or not word.isalpha() or len(word) < 3:
            continue
        # Mid-sentence capitalized names are not perturbation candidates.
        prefix = document[:match.start()].rstrip()
        if word[0].isupper() and prefix and prefix[-1] not in '.!?:\n':
            continue
        positions = (wn.ADJ,) if tag.startswith('JJ') else (wn.ADV,) if tag.startswith('RB') else ()
        for pos in positions:
            for lemma in wn.lemmas(word.lower(), pos=pos):
                # No implicit morphology: do not turn comparatives into base adjectives.
                if lemma.name().lower() != word.lower():
                    continue
                for antonym in lemma.antonyms():
                    replacement = antonym.name()
                    if not replacement.isalpha() or replacement.lower() == word.lower():
                        continue
                    if word.istitle():
                        replacement = replacement.capitalize()
                    candidate = {'position': position, 'original': word, 'replacement': replacement,
                                 'source_lemma': lemma.key(), 'antonym_lemma': antonym.key(),
                                 'sense': lemma.synset().definition(), 'pos': pos}
                    if not any(c['position'] == position and c['replacement'] == replacement for c in candidates):
                        candidates.append(candidate)
    return candidates


def select_lexical_edits(candidates, choices):
    """Only accept model-selected whitelist entries; one edit per original position."""
    selected, positions = [], set()
    if not isinstance(choices, list):
        return selected
    for choice in choices:
        if type(choice) is not int or not 0 <= choice < len(candidates):
            continue
        candidate = candidates[choice]
        if candidate['position'] in positions:
            continue
        selected.append({**candidate, 'selection': 'local_contextual_selection_from_wordnet_antonyms'})
        positions.add(candidate['position'])
    return selected


def antonym_edits(generator, document, output):
    output = Path(output)
    if output.exists():
        saved = read_json(output)
        if saved['document_sha256'] != text_hash(document):
            raise ValueError('DCMI perturbation cache identity mismatch')
        return saved
    from nltk.corpus import wordnet as wn
    candidates = lexical_candidates(document)
    evidence, collected = [], {}
    # Small selection tasks avoid both whole-document copying and long reasoning loops.
    for start in range(0, len(candidates), 16):
        group = candidates[start:start + 16]
        options = [{'id': i, 'original': c['original'], 'opposite': c['replacement'],
                    'meaning': c['sense']} for i, c in enumerate(group)]
        prompt = ('Select word substitutions that reverse an adjective/adverb meaning in THIS document '
                  'while keeping grammar natural. The available pairs are dictionary antonyms, but some '
                  'senses may be wrong in context. Reject wrong senses, nouns, verbs and named entities. '
                  'Choose every suitable distinct word position, most salient first. Do not rewrite the text. '
                  f'Return JSON only: {{"selected":[0,2]}}; use option IDs from 0 to {len(group) - 1}, '
                  'not word positions in the document. Return [] if none apply.\n'
                  'Options:\n' + json.dumps(options) + '\nDocument:\n' + document)
        for attempt in range(2):
            raw = generator.generate(prompt, 192, temperature=.5 if attempt else 0.)
            evidence.append({'prompt': prompt, 'raw': raw, 'attempt': attempt})
            try:
                parsed = json_object(raw)
            except (ValueError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue
            if not isinstance(parsed.get('selected'), list):
                continue
            accepted = select_lexical_edits(group, parsed['selected'])
            if parsed['selected'] and not accepted:
                continue
            for edit in accepted:
                collected.setdefault(edit['position'], edit)
            break
    if not collected:
        write_json(output.with_suffix('.failed.json'), {'document_sha256': text_hash(document),
                   'candidates': candidates, 'evidence': evidence})
        raise UnscorableSample('No context-selected WordNet adjective/adverb antonym available')
    saved = {'document_sha256': text_hash(document),
             'word_count': len(list(re.finditer(r"\b\w+(?:[-']\w+)*\b", document))),
             'edits': list(collected.values()), 'generation_evidence': evidence,
             'wordnet_version': wn.get_version(), 'lexical_candidates': candidates,
             'construction': 'wordnet_antonym_whitelist_with_local_context_selection',
             'pos_filter': 'averaged_perceptron_tagger_eng_adjective_adverb_only',
             'semantic_validation': 'WordNet_direct_antonym_relation_and_local_model_context_check_not_human_review'}
    write_json(output, saved)
    return saved
