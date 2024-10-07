import glob
import os
import cv2
import numpy as np
from quality_trainer.squeeze import squeeze


if __name__ == "__main__":
    input_dir = "./images/input"
    output_dir = "./images/prepared"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for dir_loop in glob.glob(os.path.join(input_dir, "*")):
        print(f"Processing directory {dir_loop}")
        dir_name = os.path.basename(dir_loop)
        for jpeg_file_loop in glob.glob(os.path.join(dir_loop, "*.jpg")):
            print(f"Processing image {jpeg_file_loop}")
            bn = os.path.basename(jpeg_file_loop)
            img = cv2.imread(jpeg_file_loop)
            img_scaled = squeeze(img)
            cv2.imwrite(os.path.join(output_dir, f"{dir_name}_{bn}"), img_scaled)
            