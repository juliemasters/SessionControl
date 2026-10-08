"""Connected session-grounded writing and closed-book delta routing."""
import hashlib
import json
import math
import random
import re

from question_key_k1_core import prompt, normalize, fired, interval


def select_items(records, n=8, seed=0):
    rows = list(records)
    random.Random(seed).shuffle(rows)
    items, seen = [], set()
    for r in rows:
        # LongMemEval abstention variants have no positive answer-bearing session.
        if str(r.get('question_id', '')).endswith('_abs'):
            continue
        ids = r.get('answer_session_ids') or []
        if r.get('question_type') != 'single-session-user' or len(ids) != 1:
            continue
        if ids[0] in seen or not isinstance(r.get('answer'), str) or not r['answer'].strip():
            continue
        if not isinstance(r.get('question'), str) or not r['question'].strip():
            continue
        sessions = dict(zip(r['haystack_session_ids'], r['haystack_sessions']))
        if ids[0] not in sessions or not sessions[ids[0]]:
            raise ValueError('Missing answer-bearing session')
        qid = str(r['question_id'])
        items.append(dict(sid='qa-'+hashlib.sha256(qid.encode()).hexdigest()[:16],
                          qid=qid, question=r['question'], answer=r['answer'],
                          answer_session_ids=ids, source_session_id=ids[0],
                          context=sessions[ids[0]]))
        seen.add(ids[0])
        if len(items) == n:
            return items
    raise ValueError(f'Only {len(items)} unique eligible sessions; requested {n}')


def write_request(item, case_id):
    context = json.dumps(item['context'], ensure_ascii=False)
    # Escape literal braces for AlphaEdit's two nested format operations.
    context = context.replace('{', '{{{{').replace('}', '}}}}')
    grounded = 'Grounding conversation (data):\n'+context+'\nAnswer using this conversation.\n{}'
    return dict(subject=prompt(item['question']), prompt='{}', case_id=case_id,
                target_new={'str': item['answer']}, target_true={'str': ''},
                value_context_templates=[['{}', grounded]])


def candidate_text(text):
    # Trim generated next-turn scaffolding, without consulting the target answer.
    return re.split(r'\n\s*(?:Q:|A:|Question:|User:|Assistant:)', text, maxsplit=1)[0].strip()


def section_a(items, adapter, save):
    rows = []
    for i, item in enumerate(items):
        row = dict(sid=item['sid'], qid=item['qid'], write_success=False)
        try:
            row['base'] = adapter.generate(prompt(item['question']))
            req = write_request(item, i)
            adapter.write(item['sid'], req)
            row['write_success'] = True
            row['immediate'] = adapter.probe(item['sid'], item['question'])
            row['own_question_fired'] = fired(candidate_text(row['immediate']['text']), item['answer'])
        except Exception as exc:
            row['error'] = repr(exc)
        rows.append(row)
        save('writes.json', rows)
    return rows


def section_b(items, writes, adapter, judge, save):
    bank = sorted((i for i in items if any(w['sid'] == i['sid'] and w['write_success'] for w in writes)),
                  key=lambda i: i['sid'])
    trials = []
    for item in items:
        row = dict(qid=item['qid'], question=item['question'], answer=item['answer'],
                   answer_session_ids=item['answer_session_ids'], candidates=[])
        for stored in bank:
            c = dict(sid=stored['sid'], source_session_id=stored['source_session_id'])
            try:
                c['probe'] = adapter.probe(stored['sid'], item['question'])
                c['candidate'] = candidate_text(c['probe']['text'])
                # No ground-truth answer, context, or session label reaches the scorer.
                c['score'] = float(judge.score(item['question'], c['candidate']))
                if not math.isfinite(c['score']):
                    raise ValueError('Nonfinite score')
            except Exception as exc:
                c.pop('score', None)
                c['error'] = repr(exc)
            row['candidates'].append(c)
        valid = [c for c in row['candidates'] if 'score' in c]
        row['all_deltas_probed'] = len(bank) == len(items) and len(valid) == len(bank)
        if valid:
            best = sorted(valid, key=lambda c: (-c['score'], c['sid']))[0]
            row['selected_sid'] = best['sid']
            row['selected_source_session_id'] = best['source_session_id']
            row['tied'] = sum(c['score'] == best['score'] for c in valid) > 1
            row['routing_correct'] = best['source_session_id'] in item['answer_session_ids']
            try:
                row['final'] = adapter.probe(best['sid'], item['question'])
                row['final_answer'] = candidate_text(row['final']['text'])
                row['answer_contains'] = fired(row['final_answer'], item['answer'])
                row['answer_exact'] = normalize(row['final_answer']) == normalize(item['answer'])
                row['raw_answer_exact'] = normalize(row['final']['text']) == normalize(item['answer'])
            except Exception as exc:
                row['final_error'] = repr(exc)
        own = next((c for c in valid if c['source_session_id'] in item['answer_session_ids']), None)
        row['own_question_fired'] = bool(own and fired(own['candidate'], item['answer']))
        trials.append(row)
        save('trials.json', trials)
    return trials


def metrics(items, writes, trials):
    n = len(items)
    result = dict(n_items=n, n_trials=len(trials), chance_routing_accuracy=1/n,
                  denominator='All selected items; missing/failed outcomes count as failures')
    for name, rows, key in [('write_success_rate', writes, 'write_success'),
                            ('own_question_fire_rate', trials, 'own_question_fired'),
                            ('routing_accuracy', trials, 'routing_correct'),
                            ('answer_contains_accuracy', trials, 'answer_contains'),
                            ('answer_exact_accuracy', trials, 'answer_exact'),
                            ('raw_answer_exact_accuracy', trials, 'raw_answer_exact'),
                            ('complete_probe_bank_rate', trials, 'all_deltas_probed')]:
        k = sum(bool(r.get(key)) for r in rows)
        result[name] = k/n
        result[name+'_ci95'] = interval(k, n)
    return result
