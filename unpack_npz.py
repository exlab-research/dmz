import numpy as np
from PIL import Image
import os
import argparse

from tqdm import tqdm


def unpack_npz(npz_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path)
    print("Keys in the .npz file:", data.files)
    for key in data.files:
        images = data[key]
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        for idx, img_array in tqdm(enumerate(images), total=len(images)):
            # Convert to uint8 and handle grayscale or RGB images
            if img_array.shape[-1] == 1:
                img_array = img_array[:, :, 0]
                mode = "L"
            else:
                mode = "RGB"
            img = Image.fromarray(np.uint8(img_array), mode=mode)

            output_path = os.path.join(output_dir, f"{idx:06}.png")
            img.save(output_path)

    print("Extraction complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Unpack an NPZ file and save images to the specified output directory."
    )
    parser.add_argument("--npz_path", type=str, help="Path to the NPZ file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the extracted images.")

    args = parser.parse_args()

    unpack_npz(args.npz_path, args.output_dir)


if __name__ == "__main__":
    main()
