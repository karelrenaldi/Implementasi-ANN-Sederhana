import sys

from src.ann import main as ann

if __name__ == "__main__":
    argv = sys.argv[1:]
    filename = argv[1]

    ann(filename)
