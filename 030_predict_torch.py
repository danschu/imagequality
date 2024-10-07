from PIL import Image
from torch import no_grad
import os
import glob
import cv2
import numpy as np
import time

from quality_trainer.squeeze import squeeze, get_default_squeeze_size
from quality_trainer.model import get_quality_model
from quality_trainer.dataloader import get_transform

def process_image(img, model, transform):
    img_small = squeeze(img)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_small = Image.fromarray(img_small)
    img_small = transform(img_small)
    img_small = img_small.cuda()
    img_small = img_small.unsqueeze(0)
    with no_grad():
        quality = model(img_small)
        quality = quality.item()*100.
    quality = int(quality)
    return quality
    
def process_dir(model_file, input_dir, output_dir):
    if not os.path.exists(model_file):
        raise Exception("Please train the model first")

    model = get_quality_model("cuda", model_file)
    img_width, img_height = get_default_squeeze_size()
    
    transform = get_transform(img_width, img_height, False)
    os.makedirs(output_dir, exist_ok=True)
     
    for jpeg_file_loop in glob.glob(os.path.join(input_dir, "*.jpg")):
        basename = os.path.basename(jpeg_file_loop)
        with open(jpeg_file_loop, "rb") as f:
            data = f.read()
                
            img = cv2.imdecode(np.asarray(bytearray(data), dtype=np.uint8), cv2.IMREAD_COLOR) 
            if img is None:
                continue
            t1 = time.time()        
            quality = process_image(img, model, transform)        
            t2 = time.time()
            
            print(jpeg_file_loop, t2-t1, quality)
            output_filepath = os.path.join(f"{output_dir}/{quality:03d}_{basename}.jpg")
            cv2.imwrite(output_filepath, img)


if __name__ == "__main__":
    model_file = "./model/image_quality_model.pth"
    input_dir = "./images/test"
    output_dir = "./images/predicted_torch"
    process_dir(model_file, input_dir, output_dir)
