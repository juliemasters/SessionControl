"""Token lookup in a grounded template, after resolving escaped literals."""
from string import Formatter


def subject_last_index(tok, template, subject):
    prefix = ''
    found = False
    for literal, field, spec, conversion in Formatter().parse(template):
        if not found:
            prefix += literal
        if field is not None:
            if found or field != '' or spec or conversion:
                raise ValueError('Require exactly one plain subject placeholder')
            found = True
    if not found:
        raise ValueError('Missing subject placeholder')
    return len(tok.encode(prefix+subject))-1
