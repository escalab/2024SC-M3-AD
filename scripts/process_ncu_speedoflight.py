import csv
profile_results = [2**i for i in range(5,19)]
print('problem,compute,memory,dram')
for l in ['m', 'n', 'k']:
    for k in profile_results:    
        profile = l+str(k)
        profile_name = profile+'.csv'
        with open(profile_name, newline='') as csvfile:
            # print(profile)
            data = csv.reader(csvfile, delimiter=',', quotechar='|')
            compute = 0
            memory = 0
            dram = 0
            count = 0
            for row in data:
                if '"Compute (SM) [%]"' in row:
                    compute += float(row[-2][1:-1])
                    # print(float(row[-2][1:-1]))
                    count += 1
                if '"Memory [%]"' in row:
                    memory += float(row[-2][1:-1])
                    # print(float(row[-2][1:-1]))
                if '"DRAM Throughput"' in row:
                    dram += float(row[-2][1:-1])
                    # print(float(row[-2][1:-1]))
            try:
                print('{},{:.3f},{:.3f},{:.3f}'.format(profile,compute/count,memory/count,dram/count))
            except:
                print(profile)
            # print('compute:',compute/count)
            # print('memory:',memory/count)
            # print('dram:',dram/count)