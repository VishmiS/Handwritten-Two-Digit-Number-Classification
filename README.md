# Handwritten-Two-Digit-Number-Classification
This code outlines the process of uploading a dataset, preprocessing it, and then training and evaluating both an sklearn classifier and a deep learning model for handwritten two-digit number classification.

## Dataset Handling: 
- We start by uploading a dataset of handwritten two-digit numbers. 
- The dataset is organized in a zip file and extracted into a directory (/content/dataset/images). 

## Data Preprocessing: 
- Images are resized to 32x32 pixels and converted into a format suitable for machine learning models (tensors). 
- Normalization is applied to standardize pixel values, ensuring consistency across images. 

## Machine Learning Approach (sklearn): 
- Using sklearn, we create a Support Vector Classifier (SVC). 
- The classifier is trained on the flattened image data from the training set. 
- Predictions are made on the test set, and accuracy is calculated to evaluate the model's performance. 

## Deep Learning Approach (TensorFlow/Keras): 
- Data is prepared using an ImageDataGenerator for both training and testing. 
- Augmentation techniques like rotation, flipping, and zooming are applied to diversify the training data. 
- A Convolutional Neural Network (CNN) model is defined: 
  - It consists of convolutional layers for feature extraction, followed by pooling layers to reduce dimensionality. 
  - Batch normalization and dropout are used to improve generalization and prevent overfitting. 
  - Dense layers with ReLU activation functions are employed for classification. 
- The model is compiled with a categorical cross-entropy loss function and the Adam optimizer. 
- It's trained using the augmented training data and validated against a subset of the training data. 
- Finally, the model's accuracy is evaluated on the test set to assess its performance. 


The goal is to achieve a testing accuracy of over 80% with the deep learning model while ensuring no overfitting. 
