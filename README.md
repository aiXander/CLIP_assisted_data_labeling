# CLIP_assisted_data_labeling
Repository to quickly label lots of images using CLIP embeddings

## Overview:
1. Create a database from a root_folder of images
2. CLIP-embed all the images in the database
3. Manually label a few images (10 minutes of labeling is usually sufficient to start)
4. Train a NN regressor/classifier on the current database (CLIP-embedding --> label)
5. Predict the labels for all the unlabeled images
6. Sort the images in one of four ways:
    - random
    - by uuid
    - best predicted first
    - worst predicted first
7. Go back to (3) and iterate until satisfied


## Detailed walkthrough:

### 0. Preprocessing your data
Best practice is to start by converting all you images to the same format eg (.jpg), using something like:

```
mogrify -format jpg *.png && rm *.png
mogrify -format jpg *.JPEG && rm *. JPEG
mogrify -format jpg *.jpeg && rm *.jpeg
mogrify -format jpg *.webp && rm *.webp
```

Metadata files (such as .txt prompt files or .npy files) that have the same basename (but different extension) as the image files can remain and will be handled correctly.


### 1. 01_rename_files
