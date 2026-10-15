"""Ground the answer target in the full source conversation."""
import json
from question_key import prompt

def write_request(item, case_id):
    context = json.dumps(item['context'], ensure_ascii=False)
    # Escape literal braces for AlphaEdit's two nested format operations.
    context = context.replace('{', '{{{{').replace('}', '}}}}')
    grounded = 'Grounding conversation (data):\n'+context+'\nAnswer using this conversation.\n{}'
    return dict(subject=prompt(item['question']), prompt='{}', case_id=case_id,
                target_new={'str': item['answer']}, target_true={'str': ''},
                value_context_templates=[['{}', grounded]])
