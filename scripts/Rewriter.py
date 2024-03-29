import re
import sys
import os

asm_load = False

def check_asm_start(line):
    global asm_load
    if "asm" in line:
        print(line)
        asm_load = True

def check_asm_end(line):
    global asm_load
    if ");" in line:
        print(line)
        asm_load = False

def print_usage():
    print(
        "usage: python3 InlineAsmAdder.py [file] [num mma insterted]"
    )

def main(argv):
    if len(argv) < 2: print_usage()
    print("Rewriting {}, add {} mma instr".format(argv[0], int(argv[1]) -1 ))
    file = open(argv[0] , "r")
    original_code = file.read()
    inline_asm = []
    with open(os.path.dirname(argv[0]) + '/new_' + os.path.basename(argv[0]), 'w') as f:
        for line in original_code.splitlines():
            f.write(line)
            f.write("\n")
            if not asm_load:
                check_asm_start(line)
            if asm_load:
                inline_asm.append(line)
                check_asm_end(line)
                if not asm_load:
                    print(inline_asm)
                    for _ in range(int(argv[1])-1):
                        for asm in inline_asm:
                            f.write(asm)
                            f.write("\n")
                    inline_asm = []
    
if __name__ == "__main__":
    main(sys.argv[1:])