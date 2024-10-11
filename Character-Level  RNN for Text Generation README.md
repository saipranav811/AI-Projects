# Character-Level RNN for Text Generation

## Overview:

This project implements a character-level Recurrent Neural Network (RNN) using PyTorch for generating text based on a given seed string.

Step-by-Step Breakdown:

### Import Libraries:

Import torch, torch.nn, and numpy for building and training the model.

### Prepare Data:

Defined a sample text dataset.
Created mappings from characters to indices and vice versa.
Generated sequences of fixed length and corresponding target characters.

### Data Encoding:

Converted sequences and targets into integer representations.
Transform data into PyTorch tensors.

### Define the RNN Model:

Created a class for the RNN model with an initialization method and a forward pass.
Use an RNN layer followed by a fully connected layer.

### Initialized Model Parameters:

Define input size (number of unique characters), hidden size, and output size.
Set Up Training:

Instantiate the model, define the loss function (CrossEntropyLoss), and choose an optimizer (Adam).

### One-Hot Encoding:

Prepared input data by one-hot encoding the character sequences.

### Train the Model:

Loop over epochs to perform forward and backward passes, calculate loss, and update model weights.

### Text Generation Function:

Defined a function to generate text by predicting characters based on a starting string.
Use one-hot encoding for input to the model during text generation.
Run Text Generation:

Set a starting string and generate new text, printing the result.

### Technologies Used:

Python
PyTorch
NumPy
