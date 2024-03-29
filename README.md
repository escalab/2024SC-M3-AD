# Micro2023-MPMXU-expr

## Microbenchmark kernels
Clone cutlass repo
```bash
$ git clone https://github.com/NVIDIA/cutlass.git
```
Follow Cutlass README steps to build CUTLASS
```bash
$ export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
$ mkdir build && cd build
$ cmake .. -DCUTLASS_NVCC_ARCHS=80 
```

Copy profiler building scripts to build directory
```bash
$ cp cutlass_profile_cmds/* cutlass/build/
```
Build cutlass profile with targeted kernels
```bash
$ cd cutlass/build/
$ chmod a+x build_profile
$ chmod a+x run_profile.sh
$ ./build_profile
$ make cutlass_profiler -j16
```
Run microbenchmark kernels
```bash
$ ./run_profile.sh
```

## Adjust Device clock frequency
Setting "coolbits" to 28 [other may work, not tested]

```bash
$ sudo nvidia-xconfig --cool-bits=28
```

**Reboot machine if following step does not work**

Enabling presistence mode (required for adjusting mem/graph clock)

```bash
$ sudo nvidia-smi -pm 1
```

List all memclk,grclk(smclk) pairs

```bash
$ nvidia-smi -q -d SUPPORTED_CLOCKS
```

Note: Common device supports few memory clock rate option while a wide range of gr/sm clock rate are supported for each memory rate.

```
        Memory                            : 9501 MHz
            Graphics                      : 2100 MHz
            Graphics                      : 2085 MHz
            Graphics                      : 2070 MHz
        ...
        Memory                            : 9251 MHz
            Graphics                      : 2100 MHz
            Graphics                      : 2085 MHz
            Graphics                      : 2070 MHz
        ...
```
Parse output with better formatting
```bash
$ nvidia-smi -q -d SUPPORTED_CLOCKS | python3 scripts/get_all_clk_mode.py
```
Adjust Graphcs clock:

```bash
$ sudo nvidia-smi -lgc [MINCLK],[MAXCLK]
```

Adjust memory clock:

```bash
$ sudo nvidia-smi -lmc [MINCLK],[MAXCLK]
```
**Note: Use actual minmum clock supported as `MINCLK`**

Monitor device clock rate:
```bash
$ watch -n 1 nvidia-smi -q -d CLOCK
```