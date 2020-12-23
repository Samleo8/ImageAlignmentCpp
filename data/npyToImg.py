import numpy as np
import sys, os
import matplotlib.pyplot as plt

def printUsage():
    print("USAGE: python npyToImg.py <npy file> [1-to-overwrite]")

if __name__ == "__main__":
    if len(sys.argv) < 1:
        printUsage()
        exit(1)

    filename = sys.argv[1]
    if not filename.endswith(".npy"):
        print("ERROR: File must be an npy file!")
        printUsage()
        exit(1)

    overwrite = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    seqName = filename.replace(".npy", "")

    if os.path.exists(seqName) and not overwrite:
        printUsage()
        raise FileExistsError(f"Folder {seqName} already exists. Use flag to force an overwrite.")
        exit()
    else:
        os.makedirs(seqName, exist_ok=True)

    imgArr = np.load(filename)