# Alien vs Predator

![alt text](https://github.com/CodecoolGlobal/alien-vs-predator-python-Przemek9110/blob/development/pic.png?raw=true)<br>

This repository contains an implementation of a multi-class image classification project. The goal is to train a model that can classify images into one of several classes. The approach used involves training two different types of Convolutional Neural Networks (CNNs): a ResNet-18 model and a custom model built from scratch.

## Goals

1. **Model Comparison**: To compare the performance of a pre-trained model (ResNet-18) with a custom model built from scratch. It is assumed that the ResNet-18 model, which has been pre-trained on a large dataset (ImageNet), should have a good generalization capability and might outperform the custom model. However, the custom model is simpler and might be faster to train and run.

2. **Understanding CNNs**: To better understand how these two different types of models behave on the same dataset and what their strengths and weaknesses are.

3. **Learning Gradio**: To gain a hands-on understanding of the Gradio library, an open-source framework for creating interactive, easy-to-use interfaces for machine learning models. The user interface created with Gradio will allow users to upload images and get model predictions with a single click.

## Approach

1. **Data Preparation**: The data used in this project primarily comes from [Kaggle](https://www.kaggle.com/datasets/pmigdal/alien-vs-predator-images). The images have already been split into training and validation sets. For the purpose of this project, a test set has been created separately. The images are augmented, transformed into tensors and normalized for optimal results during model training.

2. **Model Training**: Both the ResNet-18 model and the custom model are trained using the same training data. The models' parameters are optimized to minimize cross-entropy loss. The models are trained for a pre-defined number of epochs.

3. **Model Evaluation**: The trained models are evaluated on the test data. The primary metric used for comparison is classification accuracy.

4. **Model Comparison**: The performances of the two models are compared. This comparison is done based on their accuracy on the test set.

## Results

The results of the project will include the trained models, their accuracy scores on the test set, and a comparison between the two models.

Please note that due to the stochastic nature of neural network training, the results may vary slightly between runs.

## Installation and startup instructions

To run the project, clone the repository from GitHub.

## Authors

The project was developed by Przemyslaw Kwiecinski.