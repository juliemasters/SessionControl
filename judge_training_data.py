"""Controlled binary answerability labels; evaluation entities/templates excluded."""
import random

# Training and validation use distinct entities AND question phrasings.
SPECS=[
 ('residence','I live in {}.',['Oslo','Lima','Accra','Dakar','Bern','Perth','Kyoto','Riga'],['My residence is in which place?','Name the town where I reside.','Tell me the location of my residence.']),
 ('visit','I want to visit {}.',['Greece','Nepal','Chile','Kenya','Laos','Malta','Cuba','Fiji'],['Which destination do I hope to visit?','Name the country on my travel wish list.','Tell me the place I wish to tour.']),
 ('animal','I like {}.',['rabbits','horses','dolphins','otters','foxes','turtles','parrots','penguins'],['Name an animal I am fond of.','Which creatures appeal to me?','Tell me my animal preference.']),
 ('food','My favorite food is {}.',['sushi','curry','tacos','pasta','ramen','risotto','falafel','paella'],['Name the food I favor.','Which food tops my list?','Tell me my top food choice.']),
 ('instrument','I play {}.',['violin','flute','cello','trumpet','clarinet','harp','oboe','banjo'],['Name the musical instrument I perform on.','Which instrument am I able to perform with?','Tell me the instrument I use to make music.']),
 ('job','I am a {}.',['nurse','pilot','lawyer','dentist','plumber','librarian','architect','pharmacist'],['Name my line of work.','Which occupation earns me a living?','Tell me the profession I practice.']),
 ('color','My favorite color is {}.',['red','yellow','purple','orange','pink','white','black','brown'],['Name the color I favor.','Which hue tops my list?','Tell me my top color choice.']),
 ('book','My favorite book is {}.',['Dune','Hamlet','Beloved','Emma','Dracula','Frankenstein','Middlemarch','Persuasion'],['Name the book I favor.','Which literary title tops my list?','Tell me my top book choice.'])]

def build(split,seed=42):
    if split not in ('train','validation'):raise ValueError(split)
    rng=random.Random(seed);rows=[]
    indices=range(6) if split=='train' else range(6,8)
    qindices=range(2) if split=='train' else [2]
    wrappers=['{}','Please answer this: {}','Recall my personal information. {}','A detail about me: {}'] if split=='train' else ['{}']
    for domain,template,entities,questions in SPECS:
        for qi in qindices:
            for wrap in wrappers:
                q=wrap.format(questions[qi])
                for i in indices:
                    fact=template.format(entities[i])
                    rows.append(dict(question=q,probe=fact,label=1,domain=domain,entity=entities[i],kind='supported'))
                    # Same words with requested value absent: not answerable.
                    absent=template.format('something I cannot remember' if domain not in ('residence','visit') else 'an unspecified place')
                    rows.append(dict(question=q,probe=absent,label=0,domain=domain,entity=None,kind='missing_value'))
                    other=rng.choice([s for s in SPECS if s[0]!=domain])
                    rows.append(dict(question=q,probe=other[1].format(other[2][i]),label=0,domain=domain,entity=other[2][i],kind='wrong_relation'))
                    # Another value for the SAME relation is still relevant.
                    j=list(indices)[(list(indices).index(i)+1)%len(list(indices))]
                    rows.append(dict(question=q,probe=template.format(entities[j]),label=1,domain=domain,entity=entities[j],kind='alternate_valid_value'))
    unique={(r['question'],r['probe']):r for r in rows};rows=list(unique.values());rng.shuffle(rows)
    return rows
