## Image quality classifier

### 1. Create environment and install dependencies

Run the script
```
000_install_windows.cmd
```
This installs a virual enviornment under `./venv` and installs the requirements from the `requirements.txt`. If you are not a windows user, then you have to creat the env on you own and install e.g. `onnxruntime-gpu` instead of `onnxruntime-directml`.

### 2. Download dataset from Kaggle

Run the script
```
001_download_testdata.cmd
```
which downloads the [dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset/data) from Kaggle and stores it into `./images/input_raw/blur-dataset.zip`. You can also download it by yourself and store it there.


# For all further steps:
You have to activate the environment by calling `.\venv\scripts\activate` on commandline before running the scripts. 

### 3. Prepare the dataset

Run
```
python 010_prepare_testdata.py
```
which unzips the dataset to `./images/input_raw/` and creates three directories with images files from the dataset:

1) `./images/input/0` with blurry images (quality "0")
2) `./images/input/100` with sharp images (quality "100")
3) `./images/test` with test images (sharp and blurry images)


### 4. (Optional) Prepare your own data

If you want to train you own data you only need to copy your images into these directories. You can also use a finer classficition by creating aditional directories (eg. `./images/input/50`) with images that are only a little bit blurry.

### 5. Squeeze the images (prepare the images for the training phase)

Run
```
python 015_squeeze.py
```
which squeezes the images into a size of 480x270 px. The squeeze function copies part of the image in (almost) the original resolution and shrinks other parts of the images by resizing them. The Squeezed images are stored under `./images/prepared`.

### 6. Run the training

Run
```
python 020_train.py
```

to train the classifier. You can stop the training when the loss does not improve anymore. The best model is saved in `./model`.


### 7. Test the pytorch-model

Run
```
python 030_predict_torch.py
```

which runs the model on the test-dataset stored in `./images/test` and saves the images to `./images/predicted_torch`.


### 8. Export the model to onnx

Run
```
python 025_export_onnx.py
```

for exporting the model to onnx. The model is saved in `./model`.


### 9. Test the onnx-model

Run
```
python 031_predict_onnx.py
```

which runs the model on the test-dataset stored in `./images/test` and saves the images to `./images/predicted_onnx`.
