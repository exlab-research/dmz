import numpy as np
from PIL import Image
import os
import argparse

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Unpack an NPZ file and save images to the specified output directory."
    )
    parser.add_argument("--eval_dir", type=str, help="Path to the NPZ file.")
    parser.add_argument("--code", type=str, help="Path to the NPZ file.")
    args = parser.parse_args()

    files = os.listdir(args.eval_dir)
    data = {}
    for file in files:
        if file.startswith(f"bridge_{args.code}") and file.endswith(".npz"):
            position = file.find("idx")
            if position < 0:
                idx = 0
            else:
                idx = file.index("idx")
                idx = int(file[idx + 3 :].split("_")[0])
            data[idx] = np.load(f"{args.eval_dir}/{file}")["arr_0"]
            print(f"to merge {file}")

    idx = 0
    to_merge = []
    while idx in data:
        to_merge.append(data[idx])
        idx += len(data[idx])
    data = np.concatenate(to_merge)
    shape = "x".join([str(s) for s in data.shape])
    output_path = os.path.join(args.eval_dir, f"bridge_{args.code}_merged_{shape}.npz")
    np.savez(output_path, data)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
