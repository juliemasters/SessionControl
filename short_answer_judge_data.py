"""Independent short-candidate answerability data; no routing labels are learned.

This classifier can reject denials and incompatible answer types. It cannot tell
two plausible values for the same question apart without independent evidence.
Only source-answer sessions are used; conversation contexts are never included.
"""
from collections import Counter
import hashlib
import json
import random
import re


DENIALS = (
    "I don't know.",
    "I do not have enough information to answer that question.",
    "I cannot remember that information.",
)
EVASIONS = (
    "That depends on your situation.",
    "You should check your records or ask someone who knows.",
    "There are several possibilities, but I cannot provide a specific answer.",
)


def canonical_id(value):
    value = str(value)
    while value.endswith('_abs'):
        value = value[:-4]
    return value


def question_key(text):
    return ' '.join(re.findall(r'\w+', str(text).casefold()))


def exclusion_sets(items):
    qids, sessions, questions = set(), set(), set()
    for row in items:
        qid = row.get('qid', row.get('question_id'))
        ids = row.get('answer_session_ids') or [row.get('source_session_id')]
        if qid is None or not row.get('question') or not ids or any(x is None for x in ids):
            raise ValueError('Evaluation items must expose qid, question and answer session IDs')
        qids.add(canonical_id(qid))
        sessions.update(canonical_id(x) for x in ids)
        questions.add(question_key(row['question']))
    if not qids:
        raise ValueError('An explicit nonempty evaluation exclusion manifest is required')
    return qids, sessions, questions


def eligible_records(records, evaluation):
    blocked_qids, blocked_sessions, blocked_questions = exclusion_sets(evaluation)
    kept, counts = [], Counter()
    for row in records:
        qid = str(row.get('question_id', ''))
        ids = row.get('answer_session_ids') or []
        if (row.get('question_type') != 'single-session-user' or len(ids) != 1
                or not qid or not isinstance(row.get('answer'), str)
                or not row['answer'].strip() or not isinstance(row.get('question'), str)
                or not row['question'].strip()):
            counts['ineligible'] += 1
            continue
        if (canonical_id(qid) in blocked_qids
                or any(canonical_id(x) in blocked_sessions for x in ids)
                or question_key(row['question']) in blocked_questions):
            counts['evaluation_exclusion'] += 1
            continue
        if qid.endswith('_abs'):
            counts['abstention_variant'] += 1
            continue
        kept.append(dict(qid=canonical_id(qid), source_session_id=canonical_id(ids[0]),
                         question=row['question'].strip(), answer=row['answer'].strip()))
    # Exact duplicate rows do not receive extra sampling weight.
    kept = list({(r['qid'], r['source_session_id'], question_key(r['question']), r['answer']): r
                 for r in kept}.values())
    return kept, dict(counts)


def grouped_split(records, seed=0, validation_fraction=.2):
    """Connected components keep shared qids, sessions and question text together."""
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must lie strictly between zero and one')
    parents = list(range(len(records)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(records):
        for key in [('qid', row['qid']), ('session', row['source_session_id']),
                    ('question', question_key(row['question']))]:
            if key in seen:
                parents[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, row in enumerate(records):
        groups.setdefault(root(i), []).append(row)
    components = list(groups.values())
    if len(components) < 2:
        raise ValueError('Need at least two independent question/session groups')
    random.Random(seed).shuffle(components)
    n_validation = max(1, min(len(components)-1, round(len(components)*validation_fraction)))
    validation = [r for group in components[:n_validation] for r in group]
    train = [r for group in components[n_validation:] for r in group]
    return train, validation


def answer_type(answer):
    """Conservative type detection is used only to construct unambiguous negatives."""
    text = answer.casefold().strip()
    if re.search(r'\b(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b', text):
        return 'duration'
    if '$' in text or re.search(r'\b(?:dollars?|euros?|pounds?)\b', text):
        return 'money'
    if '%' in text or 'percent' in text:
        return 'percentage'
    if re.search(r'\b(?:am|pm|january|february|march|april|may|june|july|august|september|october|november|december)\b', text):
        return 'time'
    if re.search(r'\b(?:mbps|gbps|gb|mb)\b|\d[- ]inch', text):
        return 'measurement'
    if re.fullmatch(r'\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten', text):
        return 'count'
    return 'nominal'


def required_type(question):
    q = question.casefold()
    if q.startswith('how long') or re.search(r'how (?:much time|many hours)|screen time', q):
        return 'duration'
    if q.startswith('how many'):
        return 'count'
    if q.startswith('how old'):
        return 'age'
    if q.startswith('where'):
        return 'location'
    if q.startswith('when') or 'what time' in q:
        return 'time'
    if re.search(r'how much.*(?:spend|paid)|worth.*paid', q):
        return 'money'
    if re.search(r'what (?:speed|size)|how much ram', q):
        return 'measurement'
    if 'discount' in q:
        return 'percentage'
    if q.startswith('who') or 'name' in q or 'color' in q or q.startswith('what '):
        return 'nominal'
    return 'unknown'


INCOMPATIBLE = {
    'duration': {'money', 'nominal', 'percentage', 'measurement'},
    'count': {'money', 'nominal', 'time', 'percentage', 'measurement'},
    'age': {'money', 'nominal', 'time', 'percentage', 'measurement'},
    'location': {'money', 'duration', 'time', 'percentage', 'measurement'},
    'time': {'money', 'percentage', 'measurement'},
    'money': {'nominal', 'duration', 'time', 'percentage', 'measurement'},
    'measurement': {'nominal', 'duration', 'time', 'money'},
    'percentage': {'nominal', 'duration', 'time', 'measurement'},
    'nominal': {'duration', 'money', 'time', 'percentage', 'measurement'},
    'unknown': set(),
}


def build_examples(records, split, seed=0, wrong_type_per_question=2):
    """Donors come exclusively from the already separated split."""
    rng = random.Random(seed)
    examples = []
    for row in records:
        qtype = required_type(row['question'])
        base = dict(split=split, qid=row['qid'], source_session_id=row['source_session_id'],
                    question=row['question'], required_type=qtype)

        def add(candidate, label, kind, donor=None):
            item = dict(base, candidate=candidate, label=label, kind=kind)
            if donor is not None:
                item.update(donor_qid=donor['qid'], donor_session_id=donor['source_session_id'],
                            donor_question=donor['question'], donor_answer_type=answer_type(donor['answer']))
            examples.append(item)

        answer = row['answer']
        add(answer, 1, 'short_gold_answer')
        add('The answer is '+answer.rstrip('.')+'.', 1, 'answer_wrapper')
        add('It is '+answer.rstrip('.')+'.', 1, 'answer_wrapper')
        for denial in DENIALS:
            add(denial, 0, 'denial')
        for evasion in EVASIONS:
            add(evasion, 0, 'generic_evasion')
        donors = [d for d in records
                  if d['qid'] != row['qid'] and d['source_session_id'] != row['source_session_id']
                  and question_key(d['question']) != question_key(row['question'])
                  and answer_type(d['answer']) in INCOMPATIBLE[qtype]
                  and question_key(d['answer']) != question_key(answer)]
        rng.shuffle(donors)
        for donor in donors[:wrong_type_per_question]:
            add(donor['answer'], 0, 'wrong_answer_type', donor)
    # Contradictory duplicate labels indicate a construction bug.
    seen = {}
    for item in examples:
        key = (question_key(item['question']), item['candidate'])
        if key in seen and seen[key]['label'] != item['label']:
            raise ValueError('Conflicting labels for the same question/candidate')
        seen[key] = item
    result = list(seen.values())
    rng.shuffle(result)
    return result


def validate_splits(train, validation, evaluation):
    blocked_qids, blocked_sessions, blocked_questions = exclusion_sets(evaluation)
    split_sets = []
    for name, rows in [('train', train), ('validation', validation)]:
        qids, sessions, questions = set(), set(), set()
        for row in rows:
            qids.add(canonical_id(row['qid']))
            sessions.add(canonical_id(row['source_session_id']))
            questions.add(question_key(row['question']))
            if 'donor_qid' in row:
                qids.add(canonical_id(row['donor_qid']))
                sessions.add(canonical_id(row['donor_session_id']))
                questions.add(question_key(row['donor_question']))
        if qids & blocked_qids or sessions & blocked_sessions or questions & blocked_questions:
            raise ValueError(name+' overlaps evaluation exclusion manifest')
        if not rows or set(row['label'] for row in rows) != {0, 1}:
            raise ValueError(name+' must contain both binary classes')
        split_sets.append((qids, sessions, questions))
    if any(a & b for a, b in zip(*split_sets)):
        raise ValueError('Train and validation overlap in qid, source session or normalized question')
    return dict(evaluation_overlap=False, train_validation_overlap=False,
                excluded_qids=sorted(blocked_qids), excluded_source_session_ids=sorted(blocked_sessions),
                excluded_normalized_questions=sorted(blocked_questions),
                train_qids=sorted(split_sets[0][0]), validation_qids=sorted(split_sets[1][0]),
                train_source_session_ids=sorted(split_sets[0][1]),
                validation_source_session_ids=sorted(split_sets[1][1]))


def prepare(records, evaluation, seed=0, validation_fraction=.2):
    usable, removed = eligible_records(records, evaluation)
    train_records, validation_records = grouped_split(usable, seed, validation_fraction)
    train = build_examples(train_records, 'train', seed)
    validation = build_examples(validation_records, 'validation', seed+1)
    audit = validate_splits(train, validation, evaluation)
    audit.update(seed=seed, validation_fraction=validation_fraction,
                 eligible_question_count=len(usable), removed_counts=removed,
                 train_question_count=len(train_records), validation_question_count=len(validation_records),
                 train_example_count=len(train), validation_example_count=len(validation),
                 train_kinds=dict(Counter(r['kind'] for r in train)),
                 validation_kinds=dict(Counter(r['kind'] for r in validation)),
                 data_sha256=hashlib.sha256(json.dumps([train, validation], sort_keys=True,
                                                     ensure_ascii=False).encode()).hexdigest(),
                 label_definition='Answerability/type compatibility, not factual or session correctness',
                 exclusions='Canonical qid, answer-bearing source session and normalized question; _abs suffix stripped',
                 grouping='Connected qid/source-session/normalized-question groups, split before donor sampling',
                 limitations='Same-relation plausible wrong values are not labeled negative; no evidence identifies them.')
    return train, validation, audit
