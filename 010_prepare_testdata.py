import os
import zipfile
import glob    
import random
import cv2
import shutil
    
def prepare_data(input_archive, output_dir_archive, output_dir_prepared, test_dir, sub_dirs):
    if not os.path.exists(sub_dirs[0]):
        with zipfile.ZipFile(input_archive, 'r') as zip_ref:
            zip_ref.extractall(output_archive)
    
    bad_images_dir = os.path.join(output_dir_prepared, "0") # 0 percent
    good_images_dir = os.path.join(output_dir_prepared, "100") # 100 percent
    os.makedirs(bad_images_dir, exist_ok=True)
    os.makedirs(good_images_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    image_index = 0
    random.seed(100)
    for idx, sub_dir in enumerate(sub_dirs):     
        sub_dir_short = os.path.basename(sub_dir)
            
        for image_file in glob.glob(os.path.join(sub_dir, "*.jpg")):
            is_test = random.random() < 0.2
            image_index += 1
            new_filename = f"image_{sub_dir_short}_{image_index:04d}.jpg"
            
            if is_test:
                shutil.copyfile(image_file, os.path.join(test_dir, new_filename))
            else:
                if idx == 0: # sharp images
                    shutil.copyfile(image_file, os.path.join(good_images_dir, new_filename))
                else:
                    shutil.copyfile(image_file, os.path.join(bad_images_dir, new_filename))    
    
    

if __name__ == "__main__":
    input_archive = "./images/input_raw/blur-dataset.zip"
    output_dir_archive = "./images/input_raw"
    output_dir_prepared = "./images/input"
    test_dir = "./images/test"
    
    sub_dirs = ["sharp", "blur_dataset_scaled", "defocused_blurred", "motion_blurred"]
    sub_dirs = [os.path.join(output_dir_archive, sub_dir) for sub_dir in sub_dirs]
    
    prepare_data(input_archive, output_dir_archive, output_dir_prepared, test_dir, sub_dirs)