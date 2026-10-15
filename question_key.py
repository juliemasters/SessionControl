"""Literal question keys and lexical evaluation helpers."""
import math
import re

def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def prompt(question):
    return f'Q: {question}\nA:'


def interval(k, n):
    if not n:
        return None
    z=1.959963984540054
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h), min(1,c+h)]
