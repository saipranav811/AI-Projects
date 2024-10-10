# House Price Prediction using Feed Forward Neural Network (PyTorch)

## Overview

This project builds a Feed Forward Neural Network (FFNN) using PyTorch to predict house prices based on various features such as lot size, year built, and square footage. The dataset used is a subset of the well-known Ames Housing Dataset and includes both categorical and continuous features.

### Dataset

The dataset contains the following features:

MSSubClass: The building class

MSZoning: The general zoning classification

LotFrontage: Linear feet of street connected to the property

LotArea: Lot size in square feet

Street: Type of road access

LotShape: General shape of the property

1stFlrSF: First-floor square footage

2ndFlrSF: Second-floor square footage

Total Years: The total age of the property (calculated as current year - year built)

SalePrice: The target variable (property sale price)



#### Data Preprocessing

The following steps were taken to prepare the data:

Dropped rows with missing values to ensure clean data.

Transformed categorical features (MSSubClass, MSZoning, Street, LotShape) using Label Encoding.

Converted continuous and categorical features to tensors, a necessary step for working with PyTorch models.
Calculated Total Years to replace the YearBuilt feature.



##### Model Architecture

The FFNN architecture includes:

Embedding Layers: For handling categorical features, which are embedded into continuous vector space.

Batch Normalization: Applied to the continuous features.

Dropout Layers: To prevent overfitting by randomly setting a fraction of input units to 0 during training.

ReLU Activation: For introducing non-linearity.

Fully Connected Layers: Three fully connected layers that gradually reduce the dimensions from input to output.




###### Embedding Size for Categorical Columns

Categorical columns were embedded as follows:

MSSubClass: 15 unique values → embedding size: (15, 8)

MSZoning: 5 unique values → embedding size: (5, 3)

Street: 2 unique values → embedding size: (2, 1)

LotShape: 4 unique values → embedding size: (4, 2)



###### Model Training

The model was trained using Mean Squared Error (MSE) as the loss function and Adam Optimizer with a learning rate of 0.01.
The training process ran for 5000 epochs with periodic logging of loss.



###### Model Evaluation

The model achieved an RMSE of ~45,899 on the test set, indicating the performance on unseen data.

###### Conclusion

This project demonstrates how to build a deep learning model for house price prediction using PyTorch, leveraging embeddings for categorical data, batch normalization, and dropout to create a robust and scalable model. Despite some challenges with model accuracy, the RMSE results provide a solid starting point for further improvements and hyperparameter tuning.





