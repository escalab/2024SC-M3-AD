#!/bin/bash
profiler=./tools/profiler/cutlass_profiler
# for generating the header
$profiler --kernels=cutlass_simt_cgemm_64x64_8x3_nt_align1 --m=32 --n=32 --k=32 --output=4mma.csv --profile-iterations=1

for size in {1024,2048,4096,8192,16384}
do
    ########################    FP32   ########################
    # M3XU
    k=$(($size*4))
    $profiler --kernels=cutlass_tensorop_s1688gemm_f16_* --m=$size --n=$size --k=$k --output=4mma.csv --append=true --profile-iterations=1
    $profiler --kernels=cutlass_tensorop_s16816gemm_f16_64* --m=$size --n=$size --k=$k --output=4mma.csv --append=true --profile-iterations=1
    $profiler --kernels=cutlass_tensorop_s16816gemm_f16_128* --m=$size --n=$size --k=$k --output=4mma.csv --append=true --profile-iterations=1
    $profiler --kernels=cutlass_tensorop_s16816gemm_f16_256* --m=$size --n=$size --k=$k --output=4mma.csv --append=true --profile-iterations=1 
done

# fp16 emu thtoughput /4
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=224 --w=224 --c=3 --k=64 --r=7 --s=7  --stride_h=2 --stride_w=2 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=112 --w=112 --c=64 --k=64 --r=1 --s=1 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=112 --w=112 --c=64 --k=64 --r=3 --s=3 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=112 --w=112 --c=64 --k=256 --r=1 --s=1 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=56 --w=56 --c=256 --k=128 --r=1 --s=1 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=56 --w=56 --c=128 --k=128 --r=3 --s=3 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=56 --w=56 --c=128 --k=512 --r=1 --s=1 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=28 --w=28 --c=512 --k=256 --r=1 --s=1 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=28 --w=28 --c=256 --k=256 --r=3 --s=3 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=28 --w=28 --c=256 --k=1024 --r=1 --s=1 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=14 --w=14 --c=1024 --k=512 --r=1 --s=1 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=14 --w=14 --c=512 --k=512 --r=3 --s=3 --output=4mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s16816fprop_optimized_f16_* --n=$n --h=14 --w=14 --c=512 --k=2048 --r=1 --s=1 --output=4mma.csv --append=true --profile-iterations=1