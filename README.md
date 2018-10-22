# UNet-pytorch

## Overview
This is the code for kaggle 2018 data science bowl nuclei segmentation (https://www.kaggle.com/c/data-science-bowl-2018). We will use UNet to perform the segmentation task.

## Dependencies

* numpy
* scipy
* tqdm
* pillow
* scikit-image
* pytorch
* pandas


## Usage

1. Download the dataset from Kaggle (https://www.kaggle.com/c/data-science-bowl-2018/data).

2. Create two folders called combined and testing_data. Run script utils.py to prepare training image and testing image, the prepared image will be inside combined and testing_data folder.

3. In class Option under script utils.py, set is_train = True and adjust three dirs and other parameters.

4. Run script train.py. The model will be saved under folder checkpoint.

5. When making prediction using testing data, set train=False in utils.py, and run script train.py again. The prediction masks will be saved to the folder specified in Option class under utils.py, and the run-length-encoding csv file will be saved in current folder.

## Training results
### U-Net Architecture
![image1](https://github.com/limingwu8/UNet-pytorch/blob/master/images/model.png)

### Some examples of prediction masks
![image2](https://github.com/limingwu8/UNet-pytorch/blob/master/images/prediction_results01.png)

### Evaluation
![image3](https://github.com/limingwu8/UNet-pytorch/blob/master/images/loss.png)