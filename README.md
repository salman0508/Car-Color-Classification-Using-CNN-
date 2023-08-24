# Car-Color-Classification-Using-CNN-
This repository contains code to train a Convolutional Neural Network (CNN) model for car color identification. The model is trained to predict the color of a car based on an input image. The project is implemented in Python using the TensorFlow library.

Introduction
This project aims to demonstrate how to train a CNN model for car color identification. It involves preprocessing a dataset of car images, training the model, and making predictions on new images. The code is organized into different sections for dataset preprocessing, model creation, training, and prediction.

Prerequisites
Before running the code, ensure you have the following:
Python (3.x recommended)
TensorFlow library (pip install tensorflow)
OpenCV library (pip install opencv-python)
NumPy library (pip install numpy)
Dataset of car images categorized by color (train, test, val)

Usage
Clone this repository to your local machine.
Organize your dataset:
Place your car images dataset in separate folders for train, test, and validation (val). Each folder should have subfolders corresponding to different car colors (e.g., "Black", "Blue", etc.).

Replace dataset paths:
In the provided Python code (car_color_identification.py), replace the train_path, test_path, and val_path variables with the actual paths to your dataset folders.

Run the code:
Open the code in a Python IDE (e.g., PyCharm) and run it. The code will preprocess the images, create a CNN model, train the model, and save it after each epoch. The model's performance on the test dataset will be printed for each epoch.

Make predictions:
To make predictions on new images, replace the new_image_path variable with the path to the image you want to predict. Run the prediction code, and it will output the predicted car color and the probability distribution over different colors.

File Structure
car_color_identification.py: The main Python script containing the code for dataset preprocessing, model creation, training, and prediction.
car_model_epochXX.h5: Saved models after each epoch during training.
testcar/grey2.jpg: Sample new image for making predictions.

Results
The code will train a CNN model to predict the color of a car based on input images. After training, the model can accurately predict car colors from new images. The accuracy and loss values during training can be observed to assess the model's performance.

Web Application Structure
The web application uses HTML to create the user interface and display prediction results. The Flask framework is employed to handle the server-side logic. CSS styles are applied to enhance the visual appearance of the interface.

app.py: The Flask application script that handles routing, file upload, and prediction logic.
templates/index.html: The HTML template for the web interface, where users can upload images and view prediction results.
static/css/styles.css: The CSS stylesheet that defines the styling of the web interface.
