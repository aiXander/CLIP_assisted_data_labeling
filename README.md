# CLIP assisted data labeling
Python toolkit to quickly label/filter lots of images using CLIP embeddings + active learning.
Main use-case is to filter large image datasets that contain lots of bad images you don't want to train on.
This kit is meant to be used on a localhost linux desktop with display (labeling is done with simple OpenCV display).

## Overview:
0. (optional, but recommended) Create unique uuid's for each img in the root_dir
1. Embed all the images in the database using CLIP
2. (optional) Remove potential duplicate images based on a cosine-similarity threshold in CLIP-space
3. Manually label a few images (10 minutes of labeling is usually sufficient to start)
Labeling supports ordering the images in several different ways:
    - by uuid (= random)
    - best predicted first
    - worst predicted first
    - median predicted first (start labeling where there's most uncertainty)
    - diversity sort (tries to start with an as diverse as possible subset of the data)
4. Train a NN regressor/classifier on the current database (CLIP-embedding --> label)
5. Predict the labels for all the unlabeled images	
	--> Go back to (3) and iterate until satisfied with the predicted labels
6. Filter your dataset using the predicted labels

## Example usage:

```
export ROOT_DIR="path_to_your_img_dir"
python _0_prep_dataset.py --root_dir $ROOT_DIR --mode rename
python _1_embed_with_CLIP.py --root_dir $ROOT_DIR
```
Optional Step:
```
python _2_remove_duplicates.py --root_dir $ROOT_DIR --mode move
```
```
python _3_label_images.py --root_dir $ROOT_DIR
python _4_train_model.py --train_data_dir path_to_labeled_root_dir --train_data_names labeled_subfolder_name_01 labeled_subfolder_name_02  --model_name model_01 --test_fraction 0.20
python _5_predict_labels.py --root_dir $ROOT_DIR --model_file name_of_trained_model_01 --copy_imgs_fraction 1.0 --batch_size 6
```
Finally, apply the trained model to a new dataset:
```
python _1_embed_with_CLIP.py --root_dir path_to_large_unlabeled_img_dir
python _6_create_subset.py --input_dir path_to_large_unlabeled_img_dir --min_score 0.4 --extensions .jpg .json
```

## Detailed walkthrough:

### 0. Preprocessing your data
Recommended to use the --convert_imgs_to_jpg flag to auto-convert all your images to .jpg
Huge images will also be auto-resized (control max_res with --max_n_pixels flag)
Metadata files (such as .txt prompt files or .json files) that have the same basename (but different extension) as the image files can remain and will be handled correctly.

In all following scripts, the root_dir is the main directory where your training images live.
Most scripts should also work if this root_dir has subfolders with eg different subsets of the data.

### _0_prep_dataset.py
The labels are kept in a .csv file with a unique identifier linking to each image.
To be sure to avoid name clashes, it is highly recommended to rename each img (and any metadata files with the same name) with a unique uuid

### _1_embed_with_CLIP.py
Specify a specific CLIP model name and a batch size and embed all images using CLIP (embeddings are stored on disk as .pt files)
For each image, 4 crops are taken:
	- square crop at the centre of the image
	- padded image to a full square
	- subcrop1 (to detect blurry images and zoomed-in details)
	- subcrop2 (to detect blurry images and zoomed-in details)
	
Additionally, some manually engineered img features are also computed and saved to disk.
	
### _2_remove_duplicates.py
Specify a cosine-similarity threshold and remove duplicate images from the dataset.
This currently only works on max ~10k imgs at a time (due to the quadratic memory requirement of the all-to-all distance matrix)
but the script randomly shuffles all imgs, so if you run this a few times that should get most of the duplicates!

### _3_label_images.py
This script currently only works on a single image folder with no subfolders!
Super basic labeling interface using opencv that support re-ordering the images based on predicted labels
Label an image using they numkeys [0-9] on the keyboard
Go forward and backwards using the arrow keys <-- / -->
If a --filename--.txt file is found, the text in it will be displayed as prompt for the img.

### _4_train_model.py
Train a simple 3-layer FC-neural network with ReLu's based on the flattened CLIP-crop embeddings to regress / classify the image labels
Flow:
	- first optimize hyperparameters using eg `--test_fraction 0.15 --n_epochs 100` and `--dont_save`
	- look at the train/test loss curves to figure out the best amount of epochs to train
	- finally do a final training run using all the data using `--test_fraction 0.0`

### _5_predict_labels.py
Predict the labels for the entire image dataset using the trained model: `--model_file name_of_your_model`
`--copy_named_imgs_fraction` can be used to see a sneak peak of labeled results in a tmp output_directory

### _6_create_subset.py
Finally, use the predicted labels to copy a subset of the dataset to an output folder.
(Currently does not yet work on folders with subdirs)

## TODO
- add requirements.txt
- add a keyboard mapping class to the labeling code that has different mappings depending on which OS is running the code (currently, the keys are hardcoded for Ubuntu 20.04)
- CLIP features are great for semantic labeling/filtering, but tend to ignore low-level details like texture sharpness, pixel grain and bluriness.
The pipeline can probably be improved by adding additional features (lpips, vgg, ...)
- Currently the scoring model is just a heavily regularized 3-layer FC-neural network. It's likely that adding a more linear component (eg SVM) could make the predictions more robust
- The labeling tool currently only supports numerical labels and the pipeline is built for regression. This could be easily extended to class labels + classification.

