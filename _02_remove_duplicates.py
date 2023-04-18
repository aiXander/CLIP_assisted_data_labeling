import os
import shutil
import torch
import argparse
from tqdm import tqdm

def get_paths_and_embeddings(root_dir, crop_to_use = 'square_padded_crop'):
    for subdir, dirs, files in os.walk(root_dir):
        print(f"Parsing {subdir}, subdirs: {dirs}, n_files: {len(files)}..")
        paths, embeddings = [], []         

        # Get all the unique filenames (without the extension) and store a list of present extensions for each one:
        unique_filenames = {}
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if filename not in unique_filenames:
                unique_filenames[filename] = []
            unique_filenames[filename].append(file_extension)

        for filename in tqdm(unique_filenames.keys()):
            extension_list = unique_filenames[filename]
            if '.jpg' in extension_list and '.pt' in extension_list:
                path = os.path.join(subdir, filename + '.jpg')
                embedding_dict = torch.load(os.path.join(subdir, filename + '.pt'))
                embedding = embedding_dict[crop_to_use].squeeze()
                paths.append(path)
                embeddings.append(embedding)

        yield paths, embeddings


def find_near_duplicates(root_dir, 
                         threshold=0.975, 
                         sim_type='cosine', # ['cosine', 'euclidean']
                         mode='copy',   # ['rename', 'copy']
                         crop_to_use = 'square_padded_crop',  # which crop CLIP embedding do we compute the similarity on?
                         ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for paths, embeddings in get_paths_and_embeddings(root_dir, crop_to_use):

        if len(paths) == 0 or len(embeddings) == 0:
            continue

        embeddings = torch.stack(embeddings).to(device)
        print("Embeddings shape:", embeddings.shape)
        print("path len: ", len(paths))
        
        # Normalize the embeddings:
        normalized_embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

        # Compute the similarity matrix:
        if sim_type == 'cosine':
            similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
        elif sim_type == 'euclidean':
            similarity_matrix = torch.cdist(normalized_embeddings, normalized_embeddings)

        # Find the near duplicates using torch.where:
        near_duplicate_indices = torch.where(torch.triu(similarity_matrix, diagonal=1) > threshold)
        # convert indices to lists of integers:
        near_duplicate_indices = list(zip(near_duplicate_indices[0].tolist(), near_duplicate_indices[1].tolist()))

        near_duplicates = [(paths[i], paths[j]) for i, j in near_duplicate_indices]

        # Get the actual similarity_value for each duplicate pair:
        near_duplicate_values = [similarity_matrix[i, j].item() for i, j in near_duplicate_indices]

        # Create a folder for the near duplicates next to the root_dir:
        output_dir = os.path.join(os.path.dirname(root_dir), f"near_duplicates_{sim_type}_{threshold}")
        os.makedirs(output_dir, exist_ok=True)

        i = 0
        for i, (img_paths, sim_value) in enumerate(zip(near_duplicates, near_duplicate_values)):
            fix_duplicate(i, img_paths, output_dir, sim_value, mode=mode)

        if mode == 'move':
            print(f"Moved {i} duplicates (out of {len(paths)} total imgs) to {output_dir}")
        elif mode == 'copy':
            print(f"Found {i} duplicates (not removed from data yet!) out of {len(paths)} total imgs, results shown in {output_dir}")


def fix_duplicate(duplicate_index, img_paths, outdir, sim_value, mode = 'copy'):
    dirname = os.path.dirname(img_paths[0])
    # get the two basenames without extensions:
    basename1 = os.path.splitext(os.path.basename(img_paths[0]))[0]
    basename2 = os.path.splitext(os.path.basename(img_paths[1]))[0]

    # find all files with this same basename:
    files1 = [os.path.join(dirname, f) for f in os.listdir(os.path.dirname(img_paths[0])) if basename1 in f]
    files2 = [os.path.join(dirname, f) for f in os.listdir(os.path.dirname(img_paths[1])) if basename2 in f]

    # copy all files to the output directory:
    for f in files1:
        if mode == 'copy':
            shutil.copy(f, os.path.join(outdir, f"{sim_value:.3f}_{duplicate_index:08d}_source_{os.path.basename(f)}"))

    for f in files2:
        if mode == 'copy':
            shutil.copy(f, os.path.join(outdir, f"{sim_value:.3f}_{duplicate_index:08d}_target_{os.path.basename(f)}"))
        if mode == 'move':
            shutil.move(f, os.path.join(outdir, f"{sim_value:.3f}_{duplicate_index:08d}_target_{os.path.basename(f)}"))

    return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='Root directory of the dataset')
    parser.add_argument('--threshold', type=float, default=0.975, help='Cosine-similarity threshold for near-duplicate detection')
    parser.add_argument('--mode', type=str, default='copy', help='copy / move, Use copy to test the script, move after')
    args = parser.parse_args()

    find_near_duplicates(root_dir=args.root_dir, threshold=args.threshold, mode=args.mode)
