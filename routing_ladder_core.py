"""Pure experiment design and evaluation; no model/GPU imports."""
import json
import math
import random
from pathlib import Path
from h2_routing import normalize, answer_correct

LADDER={
 'L0':['loc_seoul','pet_dogs','food_pizza','hobby_piano','job_teacher','travel_ice','color_blue','book_atlas'],
 'L1':['loc_seoul','travel_ice','food_pizza','book_atlas','hobby_piano','color_blue','job_teacher','pet_dogs'],
 'L2':['loc_seoul','loc_busan','pet_dogs','pet_cats','food_pizza','hobby_piano','job_teacher','color_blue'],
 'L3':['pet_dogs','pet_cats']}

def load_sessions(level,n=None):
    bank=json.loads(Path(__file__).with_name('routing_ladder_bank.json').read_text())
    by={s['sid']:s for s in bank}; ids=LADDER[level]
    n=len(ids) if n is None else n
    if not 2<=n<=len(ids):raise ValueError('Session count must be 2..size of level')
    rows=[by[s] for s in ids[:n]]
    # Avoid broad preference prompts across food, pets, colors and books.
    for s in rows:
        if s['domain']=='pet':
            s['questions']=['What animals do I like?','Which animals am I fond of?',
                'What kind of animal do I like?','Which animals do I prefer?','What animals appeal to me?']
    return rows

def trials(sessions,seed=42):
    rng=random.Random(seed);result=[]
    for s in sessions:
        compatible=[x['sid'] for x in sessions if (x['domain'],x['relation'])==(s['domain'],s['relation'])]
        others=[x['sid'] for x in sessions if x['sid']!=s['sid']];rng.shuffle(others)
        for i,q in enumerate(s['questions']):
            for cohort,active in [('inactive',others[i%len(others)]),('active_control',s['sid'])]:
                result.append(dict(target=s['sid'],active=active,question=q,cohort=cohort,
                    qid=f"{s['sid']}-{i}",compatible=compatible,ambiguous=len(compatible)>1,
                    aliases=[s['entity']]+(['dog'] if s['entity']=='dogs' else ['cat'] if s['entity']=='cats' else [])))
    rng.shuffle(result)
    return result

def select(scores,recency,epsilon,rng=None,tie_break='recent'):
    if not math.isfinite(epsilon) or epsilon<0:raise ValueError('Nonnegative finite epsilon required')
    if len(scores)<2 or any(not math.isfinite(x) for x in scores.values()):raise ValueError('Need finite scores for 2+ candidates')
    best=max(scores.values());band=[s for s,v in scores.items() if best-v<=epsilon or math.isclose(best-v,epsilon,abs_tol=1e-12)]
    if tie_break=='recent':winner=max(band,key=lambda s:(recency[s],s))
    elif tie_break=='random' and rng is not None:winner=rng.choice(band)
    else:raise ValueError('Random tie break needs seeded RNG')
    return winner,band

def low_confidence(scores,threshold=.55):
    if not 0<=threshold<=1:raise ValueError('Probability threshold must be in [0,1]')
    if not scores or any(not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):raise ValueError('Expected BERT probabilities')
    return max(scores.values())<threshold

def evaluate(t,scores,selected,band,answer,oracle,probe_answer):
    return dict(t,selected=selected,scores=scores,tie_candidates=band,tied=len(band)>1,
        exact_tied=sum(v==max(scores.values()) for v in scores.values())>1,
        correct=selected==t['target'],compatible_selected=selected in t['compatible'],
        margin=scores[t['target']]-max(v for s,v in scores.items() if s!=t['target']),
        final_answer=answer,answer_correct=answer_correct(answer,t['aliases']),
        oracle_answer=oracle,oracle_answer_correct=answer_correct(oracle,t['aliases']),
        probe_assisted_answer=probe_answer,probe_assisted_correct=answer_correct(probe_answer,t['aliases']))

def wilson(k,n):
    if not n:return None
    z=1.96;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,center-half),min(1,center+half)]

def summarize(rows,sids):
    def cohort(rs):
        rate=lambda key:sum(r[key] for r in rs)/len(rs) if rs else None
        active=[r for r in rs if r['active']==r['target']]
        inactive=[r for r in rs if r['active']!=r['target']]
        return {'n':len(rs),'routing_accuracy':rate('correct'),'routing_ci_descriptive':wilson(sum(r['correct'] for r in rs),len(rs)),
            'switch_accuracy':sum(r['correct'] for r in inactive)/len(inactive) if inactive else None,
            'false_switch_rate':sum(r['selected']!=r['active'] for r in active)/len(active) if active else None,
            'answer_accuracy_weights_only':rate('answer_correct'),'answer_accuracy_probe_assisted':rate('probe_assisted_correct'),
            'oracle_id_answer_accuracy_weights_only':rate('oracle_answer_correct'),'tie_rate':rate('tied'),
            'compatible_selection_rate':rate('compatible_selected'),
            'mean_target_margin':sum(r['margin'] for r in rs)/len(rs) if rs else None}
    matrix={s:{t:0 for t in sids} for s in sids}
    for r in rows:matrix[r['target']][r['selected']]+=1
    return {'all_diagnostic':cohort(rows),'unique_answerable':cohort([r for r in rows if not r['ambiguous']]),
        'ambiguous_diagnostic_only':cohort([r for r in rows if r['ambiguous']]),'confusion':matrix,
        'random_chance':1/len(sids),'oracle_id_routing_baseline':1.0,
        'ci_caveat':'Wilson intervals are descriptive; paraphrases and repeated controls are correlated.'}
