"""H2 data, trial design, tolerance ties and metrics. Standard library only."""
import math
import random
import re
from statistics import mean

# One fact per independent session. Five semantic question variants per fact.
FACTS = [
 ('I like dogs.', ['dogs','dog'], ['What animals do I like?','Which animals am I fond of?','What kind of animal do I like?','Which animals do I enjoy?','What animals appeal to me?']),
 ('I live in Seoul.', ['Seoul'], ['Where do I live?','Which city is my home?','What city do I reside in?','Where is my current home city?','In which city am I living?']),
 ('My favorite food is pizza.', ['pizza'], ['What is my favorite food?','Which food do I like best?','What food is my top choice?','Which food do I favor most?','What is my preferred food?']),
 ('I play guitar.', ['guitar','the guitar'], ['What instrument do I play?','Which musical instrument can I play?','What is the instrument I play?','Which instrument do I perform on?','What musical instrument do I use?']),
 ('I work as a chef.', ['chef','a chef'], ['What is my profession?','What do I do for a living?','Which occupation do I have?','What is my job?','What kind of work do I do?']),
 ('I commute by bicycle.', ['bicycle','bike','by bicycle','by bike'], ['How do I commute?','What transport do I use to get to work?','How do I travel to work?','What is my commuting method?','Which vehicle do I use for commuting?']),
 ('My favorite color is green.', ['green'], ['What is my favorite color?','Which color do I like best?','What color do I prefer?','Which color is my top choice?','What color appeals to me most?']),
 ('My dog is named Milo.', ['Milo'], ['What is my dog called?','What is the name of my dog?','Which name did I give my dog?','What name does my dog have?','What do I call my dog?']),
 ('I speak Italian.', ['Italian'], ['What language do I speak?','Which language can I speak?','What is the language I know?','Which language do I use to communicate?','What language am I able to speak?']),
 ('My favorite sport is tennis.', ['tennis'], ['What is my favorite sport?','Which sport do I like best?','What sport do I favor?','Which sport is my top choice?','What sport do I prefer?']),
 ('My birthday is in April.', ['April'], ['What month is my birthday?','In which month was I born?','When is my birthday month?','Which month contains my birthday?','What is my birth month?']),
 ('I drink tea each morning.', ['tea'], ['What do I drink each morning?','Which beverage do I have in the morning?','What is my morning drink?','What drink starts my day?','Which drink do I choose at breakfast?']),
 ('I grow lavender.', ['lavender'], ['What plant do I grow?','Which plant am I growing?','What do I cultivate in my garden?','Which flowering plant do I cultivate?','What is the plant I cultivate?']),
 ('My favorite planet is Saturn.', ['Saturn'], ['What is my favorite planet?','Which planet do I like best?','What planet do I favor?','Which planet is my top choice?','What planet do I prefer?']),
 ('I collect stamps.', ['stamps','stamp'], ['What do I collect?','Which items do I collect?','What is in my collection?','What objects do I collect as a hobby?','What does my collection consist of?']),
 ('My sister is named Hana.', ['Hana'], ['What is my sister called?','What is the name of my sister?','Which name does my sister have?','What do I call my sister?','What is my sister\'s name?']),
 ('I drive a Volvo.', ['Volvo','a Volvo'], ['What brand of car do I drive?','Which car manufacturer do I drive?','What make is my car?','Which brand made the car I drive?','What is the make of my vehicle?']),
 ('My favorite season is autumn.', ['autumn','fall'], ['What is my favorite season?','Which season do I like best?','What season do I prefer?','Which season is my top choice?','What season do I favor?']),
 ('I study chemistry.', ['chemistry'], ['What subject do I study?','Which academic subject am I studying?','What is my field of study?','Which discipline do I study?','What subject am I learning?']),
 ('My favorite dessert is tiramisu.', ['tiramisu'], ['What is my favorite dessert?','Which dessert do I like best?','What dessert do I prefer?','Which dessert is my top choice?','What dessert do I favor?']),
 ('I wake up at six.', ['six','6','six oclock','6 am'], ['What time do I wake up?','When do I wake each day?','At what hour do I get up?','What is my waking time?','When does my day start?']),
 ('My favorite author is Tolkien.', ['Tolkien','J R R Tolkien'], ['Who is my favorite author?','Which author do I like best?','Who is my preferred writer?','Which writer is my favorite?','What author do I favor?']),
 ('I practice yoga.', ['yoga'], ['What exercise practice do I follow?','Which exercise discipline do I practice?','What is my exercise practice?','What mind and body exercise do I do?','Which practice do I use for exercise?']),
 ('My phone is made by Samsung.', ['Samsung'], ['Who makes my phone?','What brand is my phone?','Which company manufactured my phone?','What is the manufacturer of my phone?','Which phone brand do I use?']),
 ('My favorite tree is oak.', ['oak','an oak'], ['What is my favorite tree?','Which tree do I like best?','What tree do I prefer?','Which tree is my top choice?','What kind of tree do I favor?']),
 ('I volunteer at a library.', ['library','a library','at a library'], ['Where do I volunteer?','What place do I volunteer at?','Where do I do voluntary work?','Which institution do I volunteer for?','At what place do I give my volunteer time?']),
 ('My favorite fabric is linen.', ['linen'], ['What is my favorite fabric?','Which fabric do I like best?','What fabric do I prefer?','Which fabric is my top choice?','What material do I favor for fabric?']),
 ('I bake bread on Sundays.', ['Sunday','Sundays','on Sundays'], ['When do I bake bread?','Which day do I bake bread?','What is my bread baking day?','On what day do I make bread?','Which day of the week do I reserve for baking bread?']),
 ('My favorite gemstone is emerald.', ['emerald'], ['What is my favorite gemstone?','Which gemstone do I like best?','What gemstone do I prefer?','Which gemstone is my top choice?','What precious stone do I favor?']),
 ('I knit scarves.', ['scarves','scarf'], ['What do I knit?','Which items do I make by knitting?','What is the object I knit?','What do I create with knitting?','Which garments do I knit?']),
 ('My favorite scent is sandalwood.', ['sandalwood'], ['What is my favorite scent?','Which scent do I like best?','What fragrance do I prefer?','Which fragrance is my top choice?','What smell do I favor?']),
 ('I keep my keys in a drawer.', ['drawer','a drawer','in a drawer'], ['Where do I keep my keys?','Where are my keys stored?','What place do I put my keys in?','Where do I normally leave my keys?','What is the storage place for my keys?']),
]
WRAPPERS = ['{}', 'Please remind me: {}', 'I am trying to recall a detail I shared earlier. {}',
            'Based on what I told you before, {}', 'Could you help me remember this? {}',
            'I have forgotten this detail about myself. {}', 'Thinking back to our earlier conversation, {}',
            'I would like to check something I mentioned previously. {}', 'Please use my earlier information to answer this: {}',
            'Here is a question about a personal detail I shared. {}']


def dataset(paraphrases=20):
    if not 20 <= paraphrases <= 50: raise ValueError('Use 20..50 questions per fact')
    rows = []
    for i, (fact, answers, bases) in enumerate(FACTS):
        questions = [w.format(q) for w in WRAPPERS for q in bases][:paraphrases]
        rows.append({'session_id':f'S{i+1:02d}', 'fact':fact,'answers':answers,
                     'questions':[{'id':f'S{i+1:02d}-Q{j+1:02d}','text':q} for j,q in enumerate(questions)]})
    return {'name':'h2-one-fact-synthetic-v1','paraphrases_per_fact':paraphrases,
            'provenance':'32 authored facts; five semantic question variants per fact combined with discourse wrappers. Synthetic correlated paraphrases, not a natural-dialogue benchmark.',
            'sessions':rows}


def trials(data, n, seed=42, active_controls_per_fact=5):
    sessions = data['sessions'][:n]
    if len(sessions)!=n or n<2: raise ValueError('Invalid session count')
    rng = random.Random(seed+n)
    ids = [s['session_id'] for s in sessions]
    result = []
    for s in sessions:
        control_ids = set(rng.sample(range(len(s['questions'])), active_controls_per_fact))
        for j,q in enumerate(s['questions']):
            row = {'n':n,'target':s['session_id'],'qid':q['id'],'question':q['text'],'answers':s['answers']}
            result.append(dict(row,trial_id=f'{n}-{q["id"]}-switch',cohort='inactive',active=rng.choice([x for x in ids if x!=s['session_id']])))
            if j in control_ids:
                result.append(dict(row,trial_id=f'{n}-{q["id"]}-active',cohort='active_control',active=s['session_id']))
    rng.shuffle(result)
    return result


def choose(scores, recency, epsilon=0.02):
    if not 0 <= epsilon <= 1 or not math.isfinite(epsilon): raise ValueError('epsilon must be in [0,1]')
    if not scores or any(not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):
        raise ValueError('Scores must be finite probabilities in [0,1]')
    best = max(scores.values())
    band = [s for s,v in scores.items() if best-v<=epsilon or math.isclose(best-v,epsilon,abs_tol=1e-12)]
    winner = max(sorted(band), key=lambda s:recency[s])
    return winner, band


def normalize(text): return ' '.join(re.findall(r'\w+',text.casefold()))
def answer_correct(answer, aliases): return normalize(answer) in {normalize(x) for x in aliases}


def evaluate(trial, scores, selected, tie_band, answer, oracle_answer):
    target, active = trial['target'],trial['active']
    return dict(trial,selected=selected,scores=scores,tie_candidates=tie_band,
        routing_correct=selected==target, switch_correct=(selected==target) if target!=active else None,
        false_switch=(selected!=active) if target==active else None,
        answer=answer,answer_correct=answer_correct(answer,trial['answers']),
        oracle_answer=oracle_answer,oracle_answer_correct=answer_correct(oracle_answer,trial['answers']),
        tied=len(tie_band)>1,exact_tied=sum(v==max(scores.values()) for v in scores.values())>1,
        margin=scores[target]-max(v for s,v in scores.items() if s!=target))


def summarize(rows,n):
    def rate(items,key): return mean(r[key] for r in items) if items else None
    inactive=[r for r in rows if r['target']!=r['active']]
    active=[r for r in rows if r['target']==r['active']]
    return {'n':n,'completed_trials':len(rows),'inactive_trials':len(inactive),'active_control_trials':len(active),
        'routing_accuracy':rate(rows,'routing_correct'),'switch_accuracy':rate(inactive,'switch_correct'),
        'false_switch_rate':rate(active,'false_switch'),'answer_accuracy':rate(rows,'answer_correct'),
        'tie_rate':rate(rows,'tied'),'exact_tie_rate':rate(rows,'exact_tied'),
        'mean_target_margin':rate(rows,'margin'),
        'inactive':{'routing_accuracy':rate(inactive,'routing_correct'),'answer_accuracy':rate(inactive,'answer_correct'),'tie_rate':rate(inactive,'tied')},
        'active_controls':{'routing_accuracy':rate(active,'routing_correct'),'answer_accuracy':rate(active,'answer_correct'),'tie_rate':rate(active,'tied')},
        'random_chance':{'routing_accuracy':1/n,'kind':'theoretical uniform session selection, not measured answer accuracy'},
        'oracle_id':{'routing_accuracy':1.0 if rows else None,'answer_accuracy':rate(rows,'oracle_answer_correct'),
                     'kind':'known target session, same generated memory and answer model; gold fact text never supplied'}}
