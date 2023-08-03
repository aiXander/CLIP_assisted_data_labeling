import numpy as np
from scipy.fftpack import dct
from scipy.stats import entropy
import cv2
import os

def colorfulness(numpy_img):
    # Split the image into its color channels
    (B, G, R) = cv2.split(numpy_img.astype("float"))
    
    # Compute rg = R - G
    rg = np.absolute(R - G)
    
    # Compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    
    # Compute the mean and standard deviation of both `rg` and `yb`
    (rg_mean, rg_std) = (np.mean(rg), np.std(rg))
    (yb_mean, yb_std) = (np.mean(yb), np.std(yb))
    
    # Combine the mean and standard deviations
    std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
    mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))
    
    # Compute the colorfulness metric
    colorfulness = std_root + (0.3 * mean_root)
    
    return colorfulness / 100

def image_entropy(image, _nbins = 256):
    """
    approximates information content of an image
    low entropy are usually images with a lot of flat colors
    high entropy is usually images with white/gray noise, high frequency edges etc.
    """
    histogram  = cv2.calcHist([image], [0], None, [_nbins], [0, _nbins])
    histogram /= histogram.sum()
    entropy = -np.sum(histogram * np.log2(histogram + np.finfo(float).eps))
    entropy = entropy / np.log2(_nbins)
    return entropy

def laplacian_variance(image, normalization_scale_factor=1e-4):
    """
    similar to image entropy, but more sensitive to edges, can detect blurriness
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = np.var(laplacian)
    normalized_variance = np.tanh(variance * normalization_scale_factor)
    return normalized_variance

class ImageFeaturizer():
    def __init__(self, max_n_pixels = 768*768):
        self.max_n_pixels = max_n_pixels

    def process(self, rgb_image, verbose = False):

        # resize the image to max_n_pixels:
        w,h = rgb_image.shape[:2]
        new_w, new_h = int(np.sqrt(self.max_n_pixels * w / h)), int(np.sqrt(self.max_n_pixels * h / w))
        rgb_image    = cv2.resize(rgb_image, (new_w, new_h), interpolation = cv2.INTER_AREA)
        gray_image   = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        hsv_img      = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        feature_dict = {
            'img_stat_width':        rgb_image.shape[1] / 768,
            'img_stat_height':       rgb_image.shape[0] / 768,
            'img_stat_aspect_ratio': rgb_image.shape[1] / rgb_image.shape[0],
            'img_stat_mean_color': np.mean(rgb_image) / 255,
            'img_stat_std_color':  np.std(rgb_image) / 255,
            'img_stat_mean_red':   np.mean(rgb_image[:,:,0]) / 255,
            'img_stat_mean_green': np.mean(rgb_image[:,:,1]) / 255,
            'img_stat_mean_blue':  np.mean(rgb_image[:,:,2]) / 255,
            'img_stat_std_red':    np.std(rgb_image[:,:,0]) / 255,
            'img_stat_std_green':  np.std(rgb_image[:,:,1]) / 255,
            'img_stat_std_blue':   np.std(rgb_image[:,:,2]) / 255,
            'img_stat_mean_gray':  np.mean(gray_image) / 255,
            'img_stat_std_gray':   np.std(gray_image) / 255,
            'img_stat_mean_hue':   np.mean(hsv_img[:,:,0]) / 255,
            'img_stat_mean_sat':   np.mean(hsv_img[:,:,1]) / 255,
            'img_stat_mean_val':   np.mean(hsv_img[:,:,2]) / 255,
            'img_stat_std_hue':    np.std(hsv_img[:,:,0]) / 255,
            'img_stat_std_sat':    np.std(hsv_img[:,:,1]) / 255,
            'img_stat_std_val':    np.std(hsv_img[:,:,2]) / 255,
            'img_stat_colorfulness':       colorfulness(rgb_image),
            'img_stat_image_entropy':      image_entropy(gray_image),
            'img_stat_laplacian_variance': laplacian_variance(gray_image)
        }

        if verbose:
            print("-----------------------------")
            for key, value in feature_dict.items():
                print(f'{key}: {value:.4f}')

        return feature_dict
    
if __name__ == '__main__':
    folder = "/home/rednax/SSD2TB/Fast_Datasets/SD/Labeling/datasets/todo"
    output_folder = "/home/rednax/SSD2TB/Fast_Datasets/SD/Labeling/datasets/todo_color"
    extensions = [".jpg", ".png"]

    os.makedirs(output_folder, exist_ok = True)

    # get all img_paths in folder:
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[1] in extensions:
                image_paths.append(os.path.join(root, file))

    for image_path in image_paths:
        image      = cv2.imread(image_path)
        featurizer = ImageFeaturizer()
        features   = featurizer.process(image, verbose = True)