from quality_trainer.dataloader import get_dataloader
from quality_trainer.model import get_quality_model
from quality_trainer.train import train_model
from quality_trainer.squeeze import get_default_squeeze_size
import glob
import os
import random

def get_splits(paths, val_ratio=0.4):
    train_paths = []
    val_paths = []
    for path in paths:
        if random.random() < val_ratio:
            val_paths.append(path)
        else:
            train_paths.append(path)
    return train_paths, val_paths
    
def get_image_paths(input_dir):
    return [image_path_loop for image_path_loop in glob.glob(os.path.join(input_dir, "*.jpg"))]

def start_training(input_dir, model_output_path):
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    batch_size_train = 16
    batch_size_val = 16
    device = "cuda"
    learning_rate = 0.001
    val_ratio = 0.2

    img_width, img_height = get_default_squeeze_size()

    paths = get_image_paths(input_dir)
    random.shuffle(paths)
    cnt = int(len(paths)*val_ratio)
    
    val_paths = paths[0:cnt]
    train_paths = paths[cnt:]
    
    print(f"#Images (val): {len(val_paths)}")
    print(f"#Images (train): {len(train_paths)}")
    
    train_dataloader = get_dataloader(train_paths, img_width, img_height, batch_size_train, True)
    val_dataloader = get_dataloader(val_paths, img_width, img_height, batch_size_val, False)
    
    model = get_quality_model(device)    
    train_model(model, model_output_path, train_dataloader, val_dataloader, learning_rate, device, num_epochs = 500, patience=None)


if __name__ == "__main__":
    model_output_path = "./model/image_quality_model.pth"
    input_dir = "./images/prepared"
    
    start_training(input_dir, model_output_path)