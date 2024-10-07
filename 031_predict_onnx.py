import onnxruntime
import os
import glob
import cv2
from quality_trainer.squeeze import squeeze
import time
import numpy as np


def process_image(img, sess):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    batch, c, h, w = input_shape
    
    
    img_small = squeeze(img)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_small, (w, h))
    img_small = img_small.astype(np.float32)
    
    # torch.ToTensor [https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html]
    img_small = img_small/255.
    img_small = img_small.transpose(2, 0, 1)
    
    img_small = img_small[np.newaxis, ...] # add batch
    quality = sess.run([output_name], {input_name: img_small})
    quality = int(quality[0]*100.)
    return quality
    
def process_dir(model_file, input_dir, output_dir):
    if not os.path.exists(model_file):
        raise Exception("Please train the model first and export it to onnx")
        
    providers = onnxruntime.get_available_providers()
    print(providers) # install onnxruntime-gpu oder onnxruntime-directml
    sess = onnxruntime.InferenceSession(model_file, providers=providers)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for jpeg_file_loop in glob.glob(os.path.join(input_dir, "*.jpg")):
        basename = os.path.basename(jpeg_file_loop)
        img = cv2.imread(jpeg_file_loop)
        
        t1 = time.time()        
        quality = process_image(img, sess)        
        t2 = time.time()
        
        print(jpeg_file_loop, t2-t1, quality)
        output_filepath = os.path.join(f"{output_dir}/{quality:03d}_{basename}.jpg")
        cv2.imwrite(output_filepath, img)


if __name__ == "__main__":
    model_file = "./model/image_quality_model.onnx"
    input_dir = "./images/test"
    output_dir = "./images/predicted_onnx"
    process_dir(model_file, input_dir, output_dir)




