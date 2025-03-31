# Dog Breed Classification System

This is a deep learning-based web application that can identify dog breeds from images. The user can upload pictures of different dog breeds and the system will identify the breed. Built with PyTorch and Flask, this system uses a pre-trained ResNet50 model fine-tuned on the Stanford Dog Breed dataset to classify 120 different dog breeds.

## Features

- 🐕 Classifies 120 different dog breeds
- 📊 Confidence scores for predictions
- 📝 Breed characteristics
- 🎯 Moderate accuracy (77.5%)

## Model Details:

- Architecture: ResNet50 (pre-trained on ImageNet)
- Training: Fine-tuned on Stanford Dog Breed dataset
- Input size: 224x224 pixels
- Output: 120 dog breed classes
- Training settings:
  - Batch size: 64
  - Learning rate: 0.001
  - Data augmentation: Random flips, rotations, and color jitter

## Datasets

- Kaggle Dataset (https://www.kaggle.com/datasets/miljan/stanford-dogs-dataset-traintest)
- Stanford Dataset (http://vision.stanford.edu/aditya86/ImageNetDogs/)

##Limitations and improvements

- The accuracy for breeds that look similar is not great (for example, it identifies a Boberman as a Black and Tan Coonhound in many instances).
- The model was taking a long time to train and therefore, I used the batch size of 64. You can change/optimize the batch size to improve accuracy.
- Even with a reduced batch size, the model took around 10 hours to train.
- You can add more details for each breed in the model.py page.
- You can figure out a way so that the model can correctly distinguish between similar looking breeds.