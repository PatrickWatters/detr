#!/usr/bin/python

import sys

print("expects fh, fm, fp")

def f1calc(kooft):
    fh,fm,fp = kooft
    return (2*fh) / (2*fh + fm + fp)

kooft = [float(i) for i in sys.argv[1:]]

f1 = f1calc(kooft)

print("f1 is {}%".format(round(f1*100,2)))
