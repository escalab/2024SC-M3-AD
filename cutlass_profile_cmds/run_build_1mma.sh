#!/bin/bash
profiler=./tools/profiler/cutlass_profiler
# for generating the header
$profiler --kernels=cutlass_simt_cgemm_64x64_8x3_nt_align1 --m=32 --n=32 --k=32 --output=1mma.csv --profile-iterations=1

# GEMM
for size in {1024,2048,4096,8192,16384}
do
    ########################    FP32   ########################
    # baseline fp32 simt
    $profiler --kernels=cutlass_simt_sgemm_* --m=$size --n=$size --k=$size --output=1mma.csv --append=true --profile-iterations=1
    # cutlass 3xtf32 fp32
    $profiler --kernels=cutlass_tensorop_s1688gemm_64* --m=$size --n=$size --k=$size --output=1mma.csv --append=true --profile-iterations=1
    $profiler --kernels=cutlass_tensorop_s1688gemm_128* --m=$size --n=$size --k=$size --output=1mma.csv --append=true --profile-iterations=1
    $profiler --kernels=cutlass_tensorop_s1688gemm_256* --m=$size --n=$size --k=$size --output=1mma.csv --append=true --profile-iterations=1

    ########################    FP32C  ########################
    # baseline fp32c simt
    $profiler --kernels=cutlass_simt_cgemm_* --m=$size --n=$size --k=$size --output=1mma.csv --append=true --profile-iterations=1
    # cutlass 3xtf32 fp32c
    $profiler --kernels=cutlass_tensorop_c1688gemm_* --m=$size --n=$size --k=$size --output=1mma.csv --append=true --profile-iterations=1

    ########################    FP64   ########################
    # baseline fp64 simt
    $profiler --kernels=cutlass_simt_dgemm_* --m=$size --n=$size --k=$size --output=1mma.csv  --append=true --profile-iterations=1
    # baseline fp64 tc
    $profiler --kernels=cutlass_tensorop_d884gemm_* --m=$size --n=$size --k=$size --output=1mma.csv  --append=true --profile-iterations=1
done

# Conv
n=16
########################    FP32   ########################
# baseline simt
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=224 --w=224 --c=3 --k=64 --r=7 --s=7  --stride_h=2 --stride_w=2 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=112 --w=112 --c=64 --k=64 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=112 --w=112 --c=64 --k=64 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=112 --w=112 --c=64 --k=256 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=56 --w=56 --c=256 --k=128 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=56 --w=56 --c=128 --k=128 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=56 --w=56 --c=128 --k=512 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=28 --w=28 --c=512 --k=256 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=28 --w=28 --c=256 --k=256 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=28 --w=28 --c=256 --k=1024 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=14 --w=14 --c=1024 --k=512 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=14 --w=14 --c=512 --k=512 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_sfprop_optimized_* --n=$n --h=14 --w=14 --c=512 --k=2048 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1

# prior cutlass 3xtf32 fp32
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=224 --w=224 --c=3 --k=64 --r=7 --s=7  --stride_h=2 --stride_w=2 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=112 --w=112 --c=64 --k=64 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=112 --w=112 --c=64 --k=64 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=112 --w=112 --c=64 --k=256 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=56 --w=56 --c=256 --k=128 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=56 --w=56 --c=128 --k=128 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=56 --w=56 --c=128 --k=512 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=28 --w=28 --c=512 --k=256 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=28 --w=28 --c=256 --k=256 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=28 --w=28 --c=256 --k=1024 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=14 --w=14 --c=1024 --k=512 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=14 --w=14 --c=512 --k=512 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_64* --n=$n --h=14 --w=14 --c=512 --k=2048 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1

$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=224 --w=224 --c=3 --k=64 --r=7 --s=7  --stride_h=2 --stride_w=2 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=112 --w=112 --c=64 --k=64 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=112 --w=112 --c=64 --k=64 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=112 --w=112 --c=64 --k=256 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=56 --w=56 --c=256 --k=128 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=56 --w=56 --c=128 --k=128 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=56 --w=56 --c=128 --k=512 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=28 --w=28 --c=512 --k=256 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=28 --w=28 --c=256 --k=256 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=28 --w=28 --c=256 --k=1024 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=14 --w=14 --c=1024 --k=512 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=14 --w=14 --c=512 --k=512 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_128* --n=$n --h=14 --w=14 --c=512 --k=2048 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1

$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=224 --w=224 --c=3 --k=64 --r=7 --s=7  --stride_h=2 --stride_w=2 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=112 --w=112 --c=64 --k=64 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=112 --w=112 --c=64 --k=64 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=112 --w=112 --c=64 --k=256 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=56 --w=56 --c=256 --k=128 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=56 --w=56 --c=128 --k=128 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=56 --w=56 --c=128 --k=512 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=28 --w=28 --c=512 --k=256 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=28 --w=28 --c=256 --k=256 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=28 --w=28 --c=256 --k=1024 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=14 --w=14 --c=1024 --k=512 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=14 --w=14 --c=512 --k=512 --r=3 --s=3 --output=1mma.csv --append=true --profile-iterations=1
$profiler --kernels=cutlass_tensorop_s1688fprop_optimized_256* --n=$n --h=14 --w=14 --c=512 --k=2048 --r=1 --s=1 --output=1mma.csv --append=true --profile-iterations=1

#######################    FP32C  ########################
# baseline simt
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=224 --w=224 --c=3 --k=64 --r=7 --s=7  --stride_h=2 --stride_w=2 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=112 --w=112 --c=64 --k=64 --r=1 --s=1 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=112 --w=112 --c=64 --k=64 --r=3 --s=3 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=112 --w=112 --c=64 --k=256 --r=1 --s=1 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=56 --w=56 --c=256 --k=128 --r=1 --s=1 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=56 --w=56 --c=128 --k=128 --r=3 --s=3 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=56 --w=56 --c=128 --k=512 --r=1 --s=1 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=28 --w=28 --c=512 --k=256 --r=1 --s=1 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=28 --w=28 --c=256 --k=256 --r=3 --s=3 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=28 --w=28 --c=256 --k=1024 --r=1 --s=1 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=14 --w=14 --c=1024 --k=512 --r=1 --s=1 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=14 --w=14 --c=512 --k=512 --r=3 --s=3 --output=1mma.csv  --append=true --profile-iterations=1
$profiler --kernels=cutlass_simt_cf32_cfprop_optimized_cf32_* --n=$n --h=14 --w=14 --c=512 --k=2048 --r=1 --s=1 --output=1mma.csv  --append=true --profile-iterations=1

