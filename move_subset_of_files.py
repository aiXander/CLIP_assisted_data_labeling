import argparse
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def crawl_directory(root_dir, file_extensions):
    files = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in file_extensions):
                basename = os.path.splitext(filename)[0]
                if basename not in files:
                    files[basename] = []
                files[basename].append(os.path.join(dirpath, filename))
    return files

def copy_files(files, out_dir, fraction_f):
    n_copied_samples = 0
    for basename, paths in tqdm(files.items()):
        if random.random() < fraction_f:
            n_copied_samples += 1
            for path in paths:
                dest_path = os.path.join(out_dir, os.path.relpath(path, root_dir))
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(path, dest_path)

    print(f"Copied {n_copied_samples} samples to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy a fraction of files with specified extensions to out_dir")
    parser.add_argument("--root_dir", help="Directory to crawl for files")
    parser.add_argument("--out_dir", default=None, help="Directory to copy selected files to (default: same as root_dir)")
    parser.add_argument("--fraction_f", type=float, default=0.01, help="Fraction of files to copy (default: 0.001)")
    parser.add_argument("--file_extensions", nargs="+", default=['.jpg'], help="List of file extensions to consider (default: .jpg)")
    args = parser.parse_args()

    # Removing any possible trailing / from root_dir:
    args.root_dir = str(Path(args.root_dir).resolve())

    if args.out_dir is None:
        args.out_dir = args.root_dir + f"_{args.fraction_f:.3f}_subset"

    root_dir = args.root_dir
    out_dir = args.out_dir
    fraction_f = args.fraction_f
    file_extensions = args.file_extensions

    files = crawl_directory(root_dir, file_extensions)
    copy_files(files, out_dir, fraction_f)