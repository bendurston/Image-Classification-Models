# Image-Classification-Models

## Model
Modified VGG-16 model with data augmentation.

## Dataset
Download the dataset from: https://www.kaggle.com/c/state-farm-distracted-driver-detection

## Execution
The model can be run from either main file. Results from previous execution are saved in the `main.ipynb` file.

### Adding Dataset Path
There are two ways that this can be done.

#### First Way
Either create a `.env` file in the same directory as the README and add the path to the training dataset. An example is shown in `sample.env`

#### Second Way
In either main file, replace the constant `PATH` with the path to the training dataset.

## Contents
This project contains a `src` directory. Which contains the `data_loading`, `data_preprocessing`, and `model` directories.

The `data_loading` directory will load the paths to the images.

The `data_preprocessing` directory will preprocess the images given their paths.

The `model` directory contains the model and all functions to execute, and examine the results.

## Requirements
The requirements are listed in `requirements.txt`. If you don't use a `.env` file for the dataset path, you do not need the `python-dotenv` dependency.

Thanks.
