# Pneumonia Image Detection Classifier
### Mod 4 Project by Jagandeep Singh and Albert Um


# Project
For this project, our objective is to classify chest x-ray images by first training a Convoluted Neural Network and then predict the image to have pneumonia or not.

# Table of Contents
IMG -- contains images created for EDA

# Data
The original dataset can be found [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) <br>

The original dataset directory:
```
chest_xray
└───test
│   └───NORMAL - 234 images
│   └───PNEUMONIA - 390 images
└───train
│   └───NORMAL - 1341 images
│   └───PNEUMONIA - 3877 images
└───val
    └───NORMAL - 8 images
    └───PNEUMONIA - 8 images
```


# Process
Looking at the distribution of images available per folder, we need to address 2 things:<br>
1. Have more validation images
2. Handle class imbalance

![Original_Dataset_Distribution](IMG/Original_dataset_Distribution.png)

### Have more validation images
We have moved all the validation images to the train folder.<br>
Once the classes (Normal and Pneumonia) have been balanced, we will train_test_split the images from the train folder to have our training images and validation images.

### Handle class imbalance
Due to class imbalance on the training set, we have augmented "new" NORMAL images to balance normal and pneumonia images. We have created new images by:<br>
- Blurr the original image
- Mirror the original image
- Sharpen the orginal image

<p style="text-align: center;">NORMAL Original vs NORMAL Blurred Original</p>

![Blurred_example](IMG/Blurred_example.png)

<p style="text-align: center;">NORMAL Original vs NORMAL Mirrored Original</p>

![Mirrored_example](IMG/Mirror_example.png)

<p style="text-align: center;">NORMAL Original vs NORMAL Sharpened Original</p>

![Sharpened_example](IMG/Sharpened_example.png)



The augmented dataset directory:
```
chest_xray
└───test
│   └───NORMAL - 234 images
│   └───PNEUMONIA - 390 images
└───train
    └───NORMAL - 3885 images
    └───PNEUMONIA - 3885 images
```

# Modeling

# Conclusion

# Further Steps

# Recommendations