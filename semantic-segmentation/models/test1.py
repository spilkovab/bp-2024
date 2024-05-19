import os.path as op
import sys

import torch
import numpy as np
from PIL import Image
import yaml

import matplotlib.pyplot as plt

sys.path.append('..')
import utils

def display_example_pair(image: np.ndarray, mask: np.ndarray) -> None:
    """ Visualizes input image and segmentation map. Used for visualizations.

    Args:
        image: (np.ndarray) the rgb image
        mask: (np.ndarray) the mask of the input image

    Returns:
        None
    """
    _, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    ax[1].imshow(mask)
    ax[1].axis('off')
    ax[1].set_title('Mask')


print(torch.cuda.is_available())

np.random.seed(42)

with open('/home/nb1/PycharmProjects/data/semantic-segmentation/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# data_path = op.join('..', config['DATA_PATH'])
example_image = Image.open(op.join('/..', '/yamaha_v0/train/iid000008/rgb.jpg'))
example_mask = Image.open(op.join('/..' , '/yamaha_v0/train/iid000008/labels.png'))
image_display = np.array(example_image)
mask_display = np.array(example_mask.convert('RGB'))
display_example_pair(image_display, mask_display)

model = torch.load(op.join('/home/nb1/PycharmProjects/data/semantic-segmentation/', config['LOAD_MODEL_PATH']))
model.eval();