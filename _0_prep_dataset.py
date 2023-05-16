import os
import shutil
import uuid
import argparse
from tqdm import tqdm
from PIL import Image

all_img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.JPEG', '.JPG', '.PNG', '.BMP', '.TIFF', '.TIF', '.WEBP']

def process_file(orig_path, new_path, args):
    """
    Given an orig_path and new_path:
    1. soft-load with PIL to check if the resolution is within bounds
    2. Optionally downsize the image
    3. Convert to jpg if necessary
    4. Rename or copy the file to the new_path
    """

    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    file_extension = os.path.splitext(orig_path)[1]

    is_image = file_extension in all_img_extensions
    converted, resized = 0, 0

    if is_image:
        img = Image.open(orig_path)
        width, height = img.size
        if (width * height) > args.max_n_pixels:
            new_width = int(width * args.max_n_pixels / (width * height))
            new_height = int(height * args.max_n_pixels / (width * height))
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
            if args.convert_imgs_to_jpg:
                new_path = os.path.splitext(new_path)[0] + '.jpg'
            img.save(new_path)
            resized = 1

        if args.convert_imgs_to_jpg and not resized:
            if file_extension != '.jpg':
                new_path  = os.path.splitext(new_path)[0] + '.jpg'
                img = Image.open(orig_path).convert("RGB")
                img.save(new_path, quality=95)
                os.remove(orig_path)
                converted = 1

    if not is_image or (not resized and not converted):
        if args.mode == 'rename':
            os.rename(orig_path, new_path)
        elif args.mode == 'copy':
            shutil.copy(orig_path, new_path)

    return converted, resized


def prep_dataset_directory(args):
    
    '''
    Rename all the files in the root_dir with a unique string identifier
    Optionally:
        - convert imgs to jpg
        - downsize imgs if needed

    '''

    os.makedirs(args.output_dir, exist_ok=True)
    renamed_counter, converted_counter, resized_counter, skipped = 0, 0, 0, 0
    print_verb = "Copied" if args.mode == 'copy' else "Renamed"

    for subdir, dirs, files in os.walk(args.root_dir):
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
                new_folder    = subdir.replace(args.root_dir, args.output_dir)
                orig_filename = os.path.join(subdir, filename + ext)
                new_filename  = os.path.join(new_folder, unique_id + ext)

                try:
                    converted, resized = process_file(orig_filename, new_filename, args)
                    renamed_counter   += 1
                    converted_counter += converted
                    resized_counter   += resized
                except Exception as e:
                    print(f"Error on {orig_filename}: {e}")
                    skipped += 1
                    continue
        
        print(f"{print_verb} {renamed_counter} files (converted {converted_counter}, resized {resized_counter}), skipped {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='Root directory of the dataset folder')
    parser.add_argument('--output_dir', type=str, default = None, help='Output directory')
    parser.add_argument('--mode', type=str, default='copy', help='Modes: rename (in place) or copy')
    parser.add_argument('--max_n_pixels', type=int, default=1920*1080, help='Resize when an img is larger than this')
    parser.add_argument('--convert_imgs_to_jpg', action='store_true', help='Convert all imgs to .jpg (default: False)')
    args = parser.parse_args()
    
    if args.mode == 'copy' and args.output_dir is None:
        raise ValueError("Output directory must be specified when mode is 'copy'")

    if args.output_dir is None:
        args.output_dir = args.root_dir
        args.mode = 'rename'

    if args.mode == 'rename':
        print("####### WARNING #######")
        print(f"you are about to rename / resize all the files inside {args.root_dir}, are you sure you want to do this?")
        answer = input("Type 'yes' to continue: ")
        if answer != 'yes':
            raise ValueError("Aborted")

    prep_dataset_directory(args)