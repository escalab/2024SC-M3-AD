import subprocess
import os
import sys
import random

# get current working directory
cwd = os.getcwd()

# generate sample dimensions

# SMAPLE_COUNT = 1000
# results = []
print('ref,query,dim,k,problem size,simt time(ms), tc time(ms), speedup')
# for _ in range(SMAPLE_COUNT):
    # ref = random.randint(128, 8192)
    # query = random.randint(128, ref//4)
    # k = random.randint(1, 128)
    # dim = random.randint(1, 512)
    # ref = random.randint(128, 512)
    # # query = random.randint(128, ref//4)
    # query = ref // 4
    # # k = random.randint(1, 128)
    # k = 1
    # dim = random.randint(1, 64)
    # k *= 16
    # ref *= 16
    # query *= 16
    # dim *= 16
    # if ( ref * dim + query * dim + ref * query ) * 32 > 20 * 1024 * 1024 * 1024:
    #     continue
for r in range(4, 18):
    ref = 2 ** r
    for f in range(4,r):
        query = 2 ** f
        for d in range(4, 13):
            dim = 2 ** d
            k = 16
            print('{},{},{},{},{}'.format(ref, query, dim, k, ref*query*dim*k), end=',', flush=True)
            # print('simt', end=',', flush=True)
            res_simt = subprocess.run(["./knn_cuda_simt", str(ref), str(query), str(dim), str(k)], stdout=subprocess.PIPE)
            # print('tc', end=',', flush=True)
            res_tc = subprocess.run(["./knn_cuda_tc", str(ref), str(query), str(dim), str(k)], stdout=subprocess.PIPE)
            simt_time = float(res_simt.stdout.decode('utf-8').split(',')[-1])
            tc_time = float(res_tc.stdout.decode('utf-8').split(',')[-1])
            print('{},{},{}'.format(simt_time, tc_time, float(simt_time)/float(tc_time)), flush=True)