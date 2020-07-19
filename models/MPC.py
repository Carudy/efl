import random
from copy import deepcopy

def divide_number(x):
    a = random.random()*0.5
    b = random.random()*(1-a)
    c = 1-a-b
    return x*a, x*b, x*c

def divide_dict(d):
    a, b, c = deepcopy(d), deepcopy(d), deepcopy(d)
    for key in d:
        x, y, z = divide_number(d[key])
        a[key], b[key], c[key] = x, y, z
    return a, b, c