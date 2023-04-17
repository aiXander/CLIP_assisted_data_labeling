import os
import shutil
import uuid
import argparse
from tqdm import tqdm

def rename_files(root_dir, output_folder, 
                 mode = 'copy',  # ['rename', 'copy']
                 ):
    
    '''
    Rename all the files in the root_dir with a unique string identifier
    '''

    os.makedirs(output_folder, exist_ok=True)
    counter, skipped = 0, 0
    print_verb = "Copied" if mode == 'copy' else "Renamed"

    for subdir, dirs, files in os.walk(root_dir):
        print(f"Parsing {subdir}, subdirs: {dirs}, n_files: {len(files)}..")            

        # Get all the unique filenames (without the extension) and store a list of present extensions for each one:
        unique_filenames = {}
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if filename not in unique_filenames:
                unique_filenames[filename] = []
            unique_filenames[filename].append(file_extension)

        for filename in tqdm(unique_filenames.keys()):            
            unique_id = str(uuid.uuid4().hex)
            extension_list = unique_filenames[filename]

            for ext in extension_list:
                new_folder  = subdir.replace(root_dir, output_folder)
                orig_filename = os.path.join(subdir, filename + ext)
                new_filename = os.path.join(new_folder, unique_id + ext)

                try:
                    if mode == 'rename':
                        os.rename(orig_filename, new_filename)
                    elif mode == 'copy':
                        os.makedirs(os.path.dirname(new_filename), exist_ok=True)
                        shutil.copy(orig_filename, new_filename)
                except:
                    print(f"Error on {orig_filename}")
                    skipped += 1
                    continue

            counter += 1
        
        print(f"{print_verb} {counter} files, skipped {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, default = None, help='Output directory')
    parser.add_argument('--mode', type=str, default='copy', help='Modes: rename (in place) or copy')
    args = parser.parse_args()

    
    if args.mode == 'copy' and args.output_dir is None:
        raise ValueError("Output directory must be specified when mode is 'copy'")

    if args.output_dir is None:
        args.output_dir = args.input_dir
        args.mode = 'rename'

    if args.mode == 'rename':
        print("####### WARNING #######")
        print(f"you are about to rename all the files inside {args.input_dir}, are you sure you want to do this?")
        answer = input("Type 'yes' to continue: ")
        if answer != 'yes':
            raise ValueError("Aborted")

    rename_files(args.input_dir, args.output_dir, mode = args.mode,)