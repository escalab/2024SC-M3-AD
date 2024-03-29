#!/bin/bash
# profiler=./test
# echo simt
# echo op,ref,query,dim,k,time\(ms\)
# echo -n simt,
# ./knn_cuda_simt 4096 1024 2048 64
# echo -n simt,
# ./knn_cuda_simt 8192 2048 2048 64
# echo -n simt,
# ./knn_cuda_simt 16384 4096 2048 64
# echo -n simt,
# ./knn_cuda_simt 32768 8192 2048 64
# echo -n simt,
# ./knn_cuda_simt 65536 16384 2048 64
# echo -n simt,
# ./knn_cuda_simt 131072 32768 2048 64

# echo -n tc,
# ./knn_cuda_tc 4096 1024 2048 64
# echo -n tc,
# ./knn_cuda_tc 8192 2048 2048 64
# echo -n tc,
# ./knn_cuda_tc 16384 4096 2048 64
# echo -n tc,
# ./knn_cuda_tc 32768 8192 2048 64
# echo -n tc,
# ./knn_cuda_tc 65536 16384 2048 64
# echo -n tc,
# ./knn_cuda_tc 131072 32768 2048 64
# for r in {13..17}
# do
#     rr=$((2**$r))
#     for f in {10..13}
#     do
#         ff=$((2**$f))
#         for d in {10..16}
#         do
#             dd=$((2**$d))
#             for k in {5..8}
#             do
#                 kk=$((2**$k))
#                 ./knn_cuda_simt $rr $ff $dd $kk
#             done
#         done
#     done
# done
# echo tc
echo ref,query,dim,k,time\(ms\)
for r in {13..17}
do
    rr=$((2**$r))
    for f in {10..13}
    do
        ff=$((2**$f))
        for d in {8..8}
        do
            dd=$((2**$d))
            for k in {4..4}
            do
                kk=$((2**$k))
                echo -n simt,
                ./knn_cuda_simt $rr $ff $dd 16
                echo -n tc,
                ./knn_cuda_tc $rr $ff $dd 16
            done
        done
    done
done