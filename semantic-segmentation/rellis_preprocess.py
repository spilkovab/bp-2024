import mxnet as mx
import os
import cv2
import numpy as np

def extract_images_and_masks(rec_path, output_image_folder, output_mask_folder):
    record = mx.recordio.MXRecordIO(rec_path, 'r')

    while True:
        item = record.read()
        if not item:
            break

        header, img = mx.recordio.unpack_img(item)
        label, width, height = header.label, header.width, header.height

        # Assuming labels are 0 for images and 1 for masks
        if label == 0:
            image_path = os.path.join(output_image_folder, f"{width}x{height}_{label}.jpg")
        else:
            image_path = os.path.join(output_mask_folder, f"{width}x{height}_{label}.jpg")

        img = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), 1)
        cv2.imwrite(image_path, img)

# Paths
rec_path = "your_dataset.rec"
output_image_folder = "images"
output_mask_folder = "masks"

# Create output folders if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Extract images and masks
extract_images_and_masks(rec_path, output_image_folder, output_mask_folder)

print("Dataset preprocessing completed.")
