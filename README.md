# CLIP_assisted_data_labeling
Python toolkit to quickly label/filter lots of images using CLIP embeddings + active learning.
Main use-case is to filter large imagedatasets that contain lots of bad images you don't want to train on.

## Overview:
0. Create unique uuid's for each img in the root_dir
1. Embed all the images in the database using CLIP
2. Remove potential duplicate images based on a cosine-similarity threshold
3. Manually label a few images (10 minutes of labeling is usually sufficient to start)
Labeling supports ordering the images in one of three ways:
    - by uuid (= random)
    - best predicted first
    - worst predicted first
    - median predicted first (start labeling where there's most uncertainty)
4. Train a NN regressor/classifier on the current database (CLIP-embedding --> label)
5. Predict the labels for all the unlabeled images	

	--> Go back to (3) and iterate until satisfied with the predicted labels
6. Filter your dataset using the predicted labels


## Detailed walkthrough:

### 0. Preprocessing your data
Best practice is to start by converting all your images to the same format eg (.jpg), using something like:

```
cd ../root_of_your_img_dir/
sudo apt-get install imagemagick
mogrify -format jpg *.png  && rm *.png
mogrify -format jpg *.JPEG && rm *.JPEG
mogrify -format jpg *.jpeg && rm *.jpeg
mogrify -format jpg *.webp && rm *.webp
```

Metadata files (such as .txt prompt files or .npy files) that have the same basename (but different extension) as the image files can remain and will be handled correctly.


### 00_rename_files.py
The labels are kept in a .csv file with a unique identifier linking to each image.
To be sure to avoid name clashes, it is highly recommended to rename each img (and any metadata files with the same name) with a unique uuid

### 01_embed_with_CLIP.py
Specify a specific CLIP model name and a batch size and embed all images using CLIP (embeddings are stored on disk as .pt files)
For each image, 4 crops are taken:
	- square crop at the centre of the image
	- padded image to a full square
	- subcrop1 (to detect blurry images and zoomed-in details)
	- subcrop2 (to detect blurry images and zoomed-in details)
	
### 02_remove_duplicates.py
Specify a cosine-similarity threshold and remove duplicate images from the dataset

### 03_label_images.py
Simple labeling interface using opencv that support re-ordering the images based on predicted labels

### 04_train_model.py
Train a simple FC-neural network based on the flattened CLIP-crop embeddings to regress / classify the image labels

### 05_predict_labels.py
Predict the labels for the entire image dataset using the trained model
copy_named_imgs_fraction can be used to see a sneak peak of labeled results

### 06_create_subset.py
Use the predicted labels to copy a subset of the dataset to an output folder
