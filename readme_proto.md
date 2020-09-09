# Pneumonia Image Detection Classifier
### Mod 4 Project by Jagandeep Singh and Albert Um


# Project
For this project, our objective is to classify chest x-ray images by first training a Convoluted Neural Network and then predict the image to have pneumonia or not.

# Table of Contents

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
Due to class imbalance on the training set, we have augmented "new" images to balance normal and pneumonia images. <br>

![Original_Dataset_Distribution](IMG/Original_dataset_Distribution.png)

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

