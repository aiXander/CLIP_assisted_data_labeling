import os
import shutil
from PIL import Image

def process_images(src_folder, tmp_folder):
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    for file in os.listdir(src_folder):
        if file.lower().endswith('.jpg'):
            file_path = os.path.join(src_folder, file)

            try:
                with Image.open(file_path) as img:
                    # Perform any image processing here if needed
                    print(f"Successfully opened {file}")
            except Exception as e:
                print(f"Error opening {file}: {e}")
                dest_path = os.path.join(tmp_folder, file)
                shutil.move(file_path, dest_path)
                print(f"Moved {file} to the tmp folder")

if __name__ == "__main__":
    target_folder = "/home/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/gordon/combo"
    tmp_directory = "/home/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/gordon/combo_errored"

    process_images(target_folder, tmp_directory)
