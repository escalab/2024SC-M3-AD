# Matrix Multiplication Module

## Directories and files in this folder:
### Directory Struture
```
├── Pattern
│   ├── op_in.data
│   ├── data_in.dat
│   ├── gold_data_out.dat
├── Pattern_Gen
│   ├── Gen_Pat.m
├── etc.v
├── etc_baseline.v
├── tb_etc.v
├── Data_Alloc.xlsx
├── ReadMe.md
```
### Files Discription
- **./Pattern**: Pattern for testbench
    - op_in.data: input op code
    - data_in.dat: input data pattern (inA, inB)
    - gold_data_out.dat: output pattern (ground truth)
- **./Pattern_Gen**
    - Gen_Pat.m: Pattern generation (MATLAB)
- **etc.v**: module supporting multiple sizes of matrix multiplication
- **etc_baseline.v**: baseline module for (4, 4) x (4, 4) multiplication
- **tb_etc.v**: testbench for etc.v
- **Data_Alloc.xlsx**: data allocation for input data and buffer

-- *To generate pattern, please run `Gen_Pat` in MATLAB under the folder of ./Pattern_Gen.*

## Comparison and Thoughts
### Comparison Table
|        Files       | (4, 4) x (4, 4) | (2, 4) x (4, 2) | (4, 2) x (2, 4) | Synthesis Area (μm^2)* |
|:------------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|   etc_baseline.v   |        ✓        |        X        |        X        |  369223.545943  |
|       etc.v        |        ✓        |        ✓        |        ✓        |  469794.492073  |

**\*in 130-nm technology, and clock freqency is 100 MHz**

### Some Thoughts
* Some adders in etc.v may be reused for future optimization.
* Since the critical path of etc.v is **longer** than etc_baseline.v, pipelining may be required in the future.