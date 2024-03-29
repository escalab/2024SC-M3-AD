import csv
import os
AMP_SPEED_UP = 12.919772070640818
MPMXU_SPEED_UP = 6.288807090601426
def masure_profermance(filename, pivot_point):
    rest_duration_AMP = 0
    rest_duration_Baseline = 0
    rest_duration_MPMXU = 0

    fwdgemm_duration_AMP = 0
    fwdgemm_duration_Baseline = 0
    fwdgemm_duration_MPMXU = 0

    backgemm_duration_AMP = 0
    backgemm_duration_Baseline = 0
    backgemm_duration_MPMXU = 0

    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        for row in data[1:]:
            kernel_id = int(row[0])
            kernel_name = row[2]
            kernel_duration = float(row[10])
            # print(kernel_id, kernel_name, kernel_duration)
            if 'gemm' in kernel_name:
                if kernel_id < pivot_point: 
                    fwdgemm_duration_MPMXU += kernel_duration/AMP_SPEED_UP
                    fwdgemm_duration_AMP += kernel_duration/AMP_SPEED_UP
                    fwdgemm_duration_Baseline += kernel_duration
                else: 
                    backgemm_duration_AMP += kernel_duration
                    backgemm_duration_MPMXU += kernel_duration/MPMXU_SPEED_UP
                    backgemm_duration_Baseline += kernel_duration
            else: 
                rest_duration_AMP += kernel_duration
                rest_duration_MPMXU += kernel_duration
                rest_duration_Baseline += kernel_duration
    print(os.path.splitext(filename)[0], total_duration_Baseline, total_duration_AMP, total_duration_MPMXU)

print('benchmark,baseline,AMP,MPMXU')
masure_profermance('vgg_small.csv',597)
masure_profermance('resnet_small.csv',1493)
masure_profermance('alexnet_small.csv',332)