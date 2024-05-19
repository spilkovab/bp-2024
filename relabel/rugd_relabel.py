
from PIL import Image
import numpy as np

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

color_palette_rugd = {
    0: {"color": [0, 0, 0], "name": "void"}, 
    1: {"color": [108, 64, 20], "name": "dirt"}, 
    2: {"color": [255, 229, 204], "name": "sand"}, 
    3: {"color": [0, 102, 0], "name": "grass"}, 
    4: {"color": [0, 255, 0], "name": "tree"}, 
    5: {"color": [0, 153, 153], "name": "pole"}, 
    6: {"color": [0, 128, 255], "name": "water"}, 
    7: {"color": [0, 0, 255], "name": "sky"}, 
    8: {"color": [255, 255, 0], "name": "vehicle"}, 
    9: {"color": [255, 0, 127], "name": "container/generic-object"}, 
    10: {"color": [64, 64, 64], "name": "asphalt"}, 
    11: {"color": [255, 128, 0], "name": "gravel"}, 
    12: {"color": [255, 0, 0], "name": "building"}, 
    13: {"color": [153, 76, 0], "name": "mulch"}, 
    14: {"color": [102, 102, 0], "name": "rock-bed"},
    15: {"color": [102, 0, 0], "name": "log"}, 
    16: {"color": [0, 255, 128], "name": "bicycle"}, 
    17: {"color": [204, 153, 255], "name": "person"}, 
    18: {"color": [102, 0, 204], "name": "fence"}, 
    19: {"color": [255, 153, 204], "name": "bush"}, 
    20: {"color": [0, 102, 102], "name": "sign"}, 
    21: {"color": [153, 204, 255], "name": "rock"}, 
    22: {"color": [102, 255, 255], "name": "bridge"}, 
    23: {"color": [101, 101, 11], "name": "concrete"}, 
    24: {"color": [114, 85, 47], "name": "picnic-table"} 
}

def apply_color_mapping(image_array, color_mapping):
    mapped_image = np.zeros_like(image_array)

    for old_index, new_color in color_mapping.items():
        indices = np.where(np.all(image_array == color_palette_rugd[old_index]["color"], axis=-1))
        mapped_image[indices] = new_color

    return mapped_image

#insert mapping
IDs =    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]        #original classes
Groups = [4, 6, 6, 5, 2, 7, 4, 1, 7, 7, 6, 6, 7, 3, 3, 7, 7, 7, 7, 0, 7, 3, 7, 6, 7]                       #new classes

#set original mask image path and convert to np array
img_path = "C:/bp/relabel/trail-3_00001.png"
old_img = Image.open(img_path)
img_np = np.array(old_img)

#create new mapping and remap old image
updated_palette_rellis = {old_index: color_palette_yamaha[new_group]["color"] for old_index, new_group in zip(IDs, Groups)}
mapped_image_array = apply_color_mapping(img_np, updated_palette_rellis)
new_img = Image.fromarray(mapped_image_array)




