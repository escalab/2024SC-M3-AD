import sys
for line in sys.stdin:
# with open('clkmode.txt') as f:
    # lines = f.readlines()
    # for line in lines:
    if ' Memory' in line:
        print('\nMemory:',line[line.find(': ')+2:line.rfind(' MHz')])
    if 'Graphics' in line:
        print(line[line.find(': ')+2:line.rfind(' MHz')],end=',')
print()