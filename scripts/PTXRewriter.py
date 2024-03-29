import re
import sys

MMA_PATTERN = re.compile(r"mma.*")
def print_usage():
    print(
        "usage: python3 PTXRewriter.py [.ptx] [num mma insterted]"
    )
def main(argv):
    print(MMA_PATTERN)
    if len(argv) < 2: print_usage()
    print("Rewriting {}, add {} mma instr".format(argv[0], int(argv[1]) -1 ))
    file = open(argv[0] , "r")
    original_ptx = file.read()
    with open(argv[0]+".new", 'w') as f:
        for instr in original_ptx.splitlines():
            matched = MMA_PATTERN.findall(instr)
            if matched:
                for _ in range(int(argv[1])-1):
                    f.write(instr)
                    f.write("\n")
            f.write(instr)
            f.write("\n")
    
if __name__ == "__main__":
    main(sys.argv[1:])