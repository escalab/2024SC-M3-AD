import subprocess
import os
import sys
import random

with open('out', 'w') as f:
    f.write('N,dict_m3xu,dict_cublas,match_m3xu,match_cublas\n')
    f.close()
for n in range(2, 20):
    with open('out', 'a') as f:
        N = n*10000
        res = subprocess.run(["./mrf", '../data/B1_brain.ra', '-N', str(N)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = res.stderr.decode('utf-8').split(',')
        f.write('{},'.format(res[0])) 
        for r in res[1:]: f.write('{},'.format(str(float(r)/1000)))
        f.write('\n')
        f.close()