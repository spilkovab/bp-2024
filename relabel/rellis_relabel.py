from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os.path as op

color_palette_rellis = {
    0: {"color": [0, 0, 0],  "name": "void"},
    1: {"color": [108, 64, 20],   "name": "dirt"},
    3: {"color": [0, 102, 0],   "name": "grass"},
    4: {"color": [0, 255, 0],  "name": "tree"},
    5: {"color": [0, 153, 153],  "name": "pole"},
    6: {"color": [0, 128, 255],  "name": "water"},
    7: {"color": [0, 0, 255],  "name": "sky"},
    8: {"color": [255, 255, 0],  "name": "vehicle"},
    9: {"color": [255, 0, 127],  "name": "object"},
    10: {"color": [64, 64, 64],  "name": "asphalt"},
    12: {"color": [255, 0, 0],  "name": "building"},
    15: {"color": [102, 0, 0],  "name": "log"},
    17: {"color": [204, 153, 255],  "name": "person"},
    18: {"color": [102, 0, 204],  "name": "fence"},
    19: {"color": [255, 153, 204],  "name": "bush"},
    23: {"color": [170, 170, 170],  "name": "concrete"},
    27: {"color": [41, 121, 255],  "name": "barrier"},
    31: {"color": [134, 255, 239],  "name": "puddle"},
    33: {"color": [99, 66, 34],  "name": "mud"},
    34: {"color": [110, 22, 138],  "name": "rubble"}
}    

color_palette_yamaha = {
    0: {"color": [0, 160, 0], "name": "low vegetation"},
    1: {"color": [1, 88, 255], "name": "sky"},
    2: {"color": [40, 80, 0], "name": "high vegetation"},
    3: {"color": [156, 76, 30], "name": "rought trail"},
    4: {"color": [255, 255, 255], "name": "non-traversable"},
    5: {"color": [128, 255, 0], "name": "traversable grass"},
    6: {"color": [178, 176, 153], "name": "smooth trail"},
    7: {"color": [255, 0, 0], "name": "obstacle"}
}

def apply_color_mapping(image_array, color_mapping):
    mapped_image = np.zeros_like(image_array)

    for old_index, new_color in color_mapping.items():
        indices = np.where(np.all(image_array == color_palette_rellis[old_index]["color"], axis=-1))
        mapped_image[indices] = new_color

    return mapped_image


def process_image_and_mask(image_folder, mask_folder, image_filename, mask_filename):

    image_path = op.join(image_folder, image_filename)
    mask_path = op.join(mask_folder, mask_filename)
    print(image_path)
    print(image_folder)
    old_img = Image.open(mask_path)
    img_np = np.array(old_img)

    updated_palette_rellis = {old_index: color_palette_yamaha[new_group]["color"] for old_index, new_group in zip(IDs, Groups)}
    mapped_image_array = apply_color_mapping(img_np, updated_palette_rellis)
    new_img = Image.fromarray(mapped_image_array)
    new_img.save(mask_path)

#mapping
IDs =    [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 31, 33, 34]        #original classes
Groups = [1, 6, 5, 2, 7, 4, 1, 7, 7, 6, 7, 7, 7, 7, 0, 6, 7, 3, 6, 3]                   #new classes

# Paths to images, masks, list
mask_folder = "C:/Users/238750/OneDrive - Vysoké učení technické v Brně/BP/datasety/Rellis_3D_pylon_camera_node_label_color/Rellis-3D/"
image_folder = "C:/Users/238750/OneDrive - Vysoké učení technické v Brně/BP/datasety/Rellis_3D_pylon_camera_node/Rellis-3D/"
lst_file_path = "C:/Users/238750/OneDrive - Vysoké učení technické v Brně/BP/datasety/Rellis_3D_image_split/test.lst"

with open(lst_file_path, 'r') as lst_file:
    for line in lst_file:
        image_filename, mask_filename = line.strip().split()
        process_image_and_mask(image_folder, mask_folder, image_filename, mask_filename)


