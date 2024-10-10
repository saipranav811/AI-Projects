# Diabetes Prediction using an Artificial Neural Network (ANN) in PyTorch
## Overview

In this project, I built an Artificial Neural Network (ANN) using PyTorch to predict diabetes outcomes based on the Pima Indians Diabetes Dataset. The dataset contains various medical parameters like glucose levels, BMI, age, and more to predict whether a person is likely to have diabetes (binary outcome: 0 or 1).

### Dataset

The dataset used was the Pima Indians Diabetes Database. It includes the following features:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration after a 2-hour oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skinfold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function (a score indicating the likelihood of diabetes based on family history)
Age: Age of the patient
Outcome: 0 for non-diabetic, 1 for diabetic (target variable)

#### Steps

1. Data Preprocessing
Imported the dataset and checked for null values using df.isnull().sum(). No missing values were found.
Split the data into input features (X) and the target label (y).
Used train_test_split from Scikit-learn to split the data into training and testing sets (80% training, 20% testing).

2. PyTorch Model Creation
Converted training and testing datasets into tensors using torch.FloatTensor for features and torch.LongTensor for the labels.

Defined a custom ANN class:

3. Model Training
Defined the loss function as CrossEntropyLoss and the optimizer as Adam.
Trained the model over 500 epochs with a learning rate of 0.01, printing the loss every 10 epochs.
Used Backpropagation to minimize the loss and updated the model's parameters using optimizer.step().

4. Model Evaluation
Made predictions on the test dataset and computed the confusion matrix

5. Model Performance
Accuracy score achieved was 33.12%, which suggests that the model requires further tuning or a different approach for better performance.

6. Saving the Model
Saved the trained model using torch.save() and loaded it using torch.load() to evaluate it again.

##### Results

Despite training the ANN model, the accuracy score of 33.12% and the confusion matrix indicate that the model struggles with high false positives and false negatives. Further improvements could be made by:

Tuning hyperparameters such as learning rate, hidden layers, or units.
Trying different architectures or models (e.g., CNN, RNN, or classical ML models).
Performing feature engineering or scaling for improved performance.



