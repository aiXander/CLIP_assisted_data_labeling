import torch
import os

def print_structure(data, indent=0):
    """Recursively prints the structure of nested dictionaries containing tensors."""
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}Key: {key}")
            if isinstance(value, torch.Tensor):
                print(f"{prefix}  Shape: {value.shape}, Dtype: {value.dtype}")
            elif isinstance(value, dict):
                print_structure(value, indent + 1)
            else:
                print(f"{prefix}  Type: {type(value)}")
    elif isinstance(data, torch.Tensor):
        # Handle case where the .pt file directly contains a tensor
        print(f"{prefix}Tensor Shape: {data.shape}, Dtype: {data.dtype}")
    else:
        print(f"{prefix}Type: {type(data)}")


if __name__ == "__main__":
    # --- IMPORTANT: Replace this with the actual path to your .pt file ---
    file_path = "/data/xander/Projects/cog/xander_eden_stuff/tmp/all/2b1e66dff81c4b70b54731ae08edb1b1.pt" 
    # --------------------------------------------------------------------

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please update the 'file_path' variable in the script.")
        exit()

    print(f"Loading data from: {file_path}")
    try:
        # Load the file, mapping to CPU to avoid GPU issues if saved on GPU
        loaded_data = torch.load(file_path, map_location='cpu')
        print("\n--- File Contents ---")
        print_structure(loaded_data)
        print("--------------------\n")

    except Exception as e:
        print(f"Error loading or processing file {file_path}: {e}") 