"""Pure data selection and metrics for the closed-book question-key test."""
import hashlib
import math
import random
import re

VARIANTS = {'V1': [13, 14, 15], 'V2': [7], 'V3': [14], 'V4': [13, 14, 15], 'V5': [13, 14, 15]}


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def fired(text, answer):
    return bool(answer.strip()) and answer.strip().casefold() in text.casefold()


def prompt(question):
    return f'Q: {question}\nA:'


def candidates(records, seed=0):
    records = list(records)
    random.Random(seed).shuffle(records)
    for rec in records:
        ids = rec.get('answer_session_ids') or []
        q, answer = rec.get('question'), rec.get('answer')
        if rec.get('question_type') != 'single-session-user' or len(ids) != 1:
            continue
        if not isinstance(q, str) or not isinstance(answer, str) or not normalize(answer) or fired(q, answer):
            continue
        session = next((turns for sid, turns in zip(rec['haystack_session_ids'], rec['haystack_sessions']) if sid == ids[0]), [])
        evidence = [dict(turn=i, content=t['content']) for i,t in enumerate(session)
                    if t.get('role') == 'user' and fired(t['content'], answer)]
        if not evidence:
            continue
        sid = 'k1-' + hashlib.sha256(str(rec['question_id']).encode()).hexdigest()[:16]
        yield dict(sid=sid, qid=rec['question_id'], question=q, answer=answer,
                   source_session_id=ids[0], source_turn=evidence[0]['content'], evidence=evidence)


def request(item, variant, case_id):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    key = 'My memory:' if variant == 'V4' else prompt(item['question'])
    value = item['source_turn'] if variant == 'V5' else item['answer']
    return dict(subject=key, prompt='{}', case_id=case_id, target_new={'str':value}, target_true={'str':''})


def interval(k, n):
    if not n:
        return None
    z=1.959963984540054
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h), min(1,c+h)]


def summarize(rows):
    n=len(rows)
    keys={'write_success_rate':'write_success', 'fire_rate_immediate':'fired_immediate',
          'fire_rate_delayed':'fired_delayed', 'base_fire_rate':'base_fired',
          'exact_match_rate':'exact_immediate', 'partial_match_rate':'fired_immediate'}
    result={'n_items':n, 'rate_denominator':'All selected items; write failures count as failures', 'ci_method':'Wilson 95%'}
    for metric,key in keys.items():
        k=sum(bool(r.get(key,False)) for r in rows)
        result[metric]=k/n if n else None
        result[metric+'_ci95']=interval(k,n)
    lengths=[r['immediate']['tokens'] for r in rows if 'immediate' in r]
    result['mean_probe_length']=sum(lengths)/len(lengths) if lengths else None
    result['probe_length_denominator']=len(lengths)
    return result
