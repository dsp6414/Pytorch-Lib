import os
import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import torch.nn.functional as F
from torch.autograd import Variable
import re
from functools import partial
import matplotlib.pyplot as plt

s = [x**2 for x in range(10)]
print(s)

print( [i for i in range(20) if i%2 == 0])



noprimes = [j for i in range(2, 8) for j in range(i*2, 50, i)]
primes = [x for x in range(2, 50) if x not in noprimes]
print (primes)

words = 'The quick brown fox jumps over the lazy dog'.split()
stuff = [[w.upper(), w.lower(), len(w)] for w in words]
for i in stuff:
    print(i)

stuff = map(lambda w: [w.upper(), w.lower(), len(w)], words)
for i in stuff:
    print(i)

nums = [1, 2, 3, 4]
fruit = ["Apples", "Peaches", "Pears", "Bananas"]
print( [(i, f) for i in nums for f in fruit])

print ([(i, f) for i in nums for f in fruit if f[0] == "P"])

rows = ([1, 2, 3], [10, 20, 30])
print([i for i in zip(*rows)])

foo = list(x for x in range(10))
print(foo)

foo = {x: str(x) for x in range(10)}
print(foo)

foo = [('a', 1), ('b', 2), ('c', 3)]
foo=dict(pair for pair in foo if pair[1] % 2 == 0)
print(foo)

foo = dict((x, str(x)) for x in range(10))
print(foo)

foo = set([x for x in range(10)])
print(foo)

foo = max([x for x in range(10)])
print(foo)

foo = tuple(x for x in range(10))
print(foo)

movies = ["Star Wars", "Gandhi", "Casablanca", "Shawshank Redemption", "Toy Story", "Gone with the Wind", "Citizen Kane", "It's a Wonderful Life", "The Wizard of Oz", "Gattaca", "Rear Window", "Ghostbusters", "To Kill A Mockingbird", "Good Will Hunting", "2001: A Space Odyssey", "Raiders of the Lost Ark", "Groundhog Day", "Close Encounters of the Third Kind"]
moviedates = [("Citizen Kane", 1941), ("Spirited Away", 2001), ("It's a Wonderful Life", 1946), ("Gattaca", 1997), ("No Country for Old Men", 2007), ("Rear Window", 1954), ("The Lord of the Rings: The Fellowship of the Ring", 2001), ("Groundhog Day", 1993), ("Close Encounters of the Third Kind", 1977), ("The Royal Tenenbaums", 2001),
    ("The Aviator", 2004), ("Raiders of the Lost Ark", 1981)]
gmovies = [str.upper(title) for title in movies if title.startswith("G")]
print(gmovies)

pre2k = [title for (title, year) in moviedates if year < 2000]
print(pre2k)
