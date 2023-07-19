import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_filepaths(root_dir, extension = ['.pt']):
    filepaths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(tuple(extension)):
                filepaths.append(os.path.join(root, file))
    return filepaths

def create_context_embedding(args, context_dir):

    if args.clip_models_to_use[0] != "all":
        print(f"\n----> Using clip models: {args.clip_models_to_use}")

    # Load the feature vectors from disk (uuid.pt)
    print(f"\nLoading CLIP features from disk...")

    context_clip_features = []
    context_pathnames = []
    n_samples = 0
    skips = 0

    # Load all the clip_embeddings from the context dir:
    for embedding_path in tqdm(get_filepaths(context_dir)):

        try:
            full_feature_dict = torch.load(embedding_path)

            if args.clip_models_to_use[0] == "all":
                args.clip_models_to_use = list(full_feature_dict.keys())
                print(f"\n----> Using all found clip models: {args.clip_models_to_use}")

            sample_features = []

            for clip_model_name in args.clip_models_to_use:
                feature_dict = full_feature_dict[clip_model_name]
                clip_embedding = feature_dict[args.crop_name_to_use].flatten()
                sample_features.append(clip_embedding)

            context_clip_features.append(torch.cat(sample_features, dim=0))
            context_pathnames.append(Path(embedding_path).name)
            n_samples += 1
        except Exception as e: # simply skip the sample if something goes wrong
            print(e)
            skips += 1
            continue

    print(f"Loaded {n_samples} samples from {context_dir}")
    if skips > 0:
        print(f"(skipped {skips} samples due to loading errors)..")

    context_clip_features = torch.stack(context_clip_features, dim=0).to(device).float()
    print("CLIP features for context loaded, shape: ", context_clip_features.shape)
    return torch.mean(context_clip_features, dim=0), context_pathnames


    
class topN():
    # Class to keep track of the top N most similar uuids and their corresponding distances
    def __init__(self, top_n):
        self.top_n = top_n
        self.best_img_paths = []
        self.best_distances = []

    def update(self, distance, img_path):

        if len(self.best_distances) < self.top_n:
            self.best_img_paths.append(img_path)
            self.best_distances.append(distance)
        else:
            # find the index of the img_path with the largest distance (using torch):
            idx = torch.tensor(self.best_distances).argmax().item()

            if distance < self.best_distances[idx]:
                self.best_img_paths[idx] = img_path
                self.best_distances[idx] = distance


def compute_distance(context_clip_embedding, sample_clip_embedding, similarity_measure):
    if similarity_measure == "cosine":
        return (1-torch.nn.functional.cosine_similarity(context_clip_embedding, sample_clip_embedding, dim=-1))/2
    elif similarity_measure == "l2":
        return torch.nn.functional.pairwise_distance(context_clip_embedding, sample_clip_embedding, p=2, eps=1e-06)
    else:
        raise NotImplementedError(f"Similarity measure {similarity_measure} not implemented!")

def find_similar_imgs(args, context_clip_embedding, context_pathnames):

    # Load the feature vectors from disk (uuid.pt)
    print(f"\nSearching {args.search_dir} for similar imgs. Saving results to {args.output_dir}..")

    n_samples = 0
    skips = 0

    topn = topN(args.top_n)

    # Load all the clip_embeddings from the context dir:
    for embedding_path in tqdm(get_filepaths(args.search_dir)):

        # Make sure there is a corresponding image file:
        img_path = embedding_path.replace(".pt", ".jpg")

        if not os.path.exists(img_path) or (Path(img_path).name in context_pathnames):
            continue

        try:
            full_feature_dict = torch.load(embedding_path)
            sample_features = []
            for clip_model_name in args.clip_models_to_use:
                feature_dict = full_feature_dict[clip_model_name]
                clip_embedding = feature_dict[args.crop_name_to_use].flatten()
                sample_features.append(clip_embedding)

            sample_clip_embedding = torch.cat(sample_features, dim=0)

            d = compute_distance(context_clip_embedding, sample_clip_embedding, args.similarity_measure)
            topn.update(d, img_path)
            n_samples += 1
        except Exception as e: # simply skip the sample if something goes wrong
            print(e)
            skips += 1
            continue

    print(f"Searched through {n_samples} samples from {args.search_dir}")
    if skips > 0:
        print(f"(skipped {skips} samples due to loading errors)..")

    return topn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar images between the context and search directories using pre-computed CLIP embeddings")
    parser.add_argument("--context_dir", help="Directory to learn img context from")
    parser.add_argument("--search_dir", help="Directory to find similar imgs in")
    parser.add_argument("--output_dir", default=None, help="Directory to copy selected files to (default: search_dir_similar)")
    parser.add_argument('--clip_models_to_use', metavar='S', type=str, nargs='+', default=['all'], help='Which CLIP model embeddings to use, default: use all found')
    parser.add_argument("--crop_name_to_use", default="square_padded_crop", help="From which img crop to use the CLIP embedding")
    parser.add_argument("--similarity_measure", default="l2", help="Similarity measure to use in CLIP-space (cosine or l2)")
    parser.add_argument("--top_n", default=30, type=int, help="How many similar images to find")
    args = parser.parse_args()

    # check if the context dir contains .pt files, if not, it's a root dir, loop over its subdirs:
    if not any([f.endswith(".pt") for f in os.listdir(args.context_dir)]):
        context_dirs = [os.path.join(args.context_dir, d) for d in os.listdir(args.context_dir)]
    else:
        context_dirs = [args.context_dir]

    for context_dir in context_dirs:
        # Create CLIP-embedding of the context images:
        context_clip_embedding, context_pathnames = create_context_embedding(args, context_dir)

        # Create the output dir if it doesn't exist:
        args.output_dir = os.path.join(context_dir, "_similar")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Find the n most similar images in the search dir:
        topn = find_similar_imgs(args, context_clip_embedding, context_pathnames)

        # Move the best images to the output dir:
        for i, img_path in enumerate(topn.best_img_paths):
            distance = topn.best_distances[i]
            orig_stem = Path(img_path).stem
            out_path = os.path.join(args.output_dir, f"{distance:.3f}_{orig_stem}.jpg")
            shutil.copy(img_path, out_path)