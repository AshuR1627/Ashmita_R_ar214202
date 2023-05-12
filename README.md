# This project is all about detecting the presence of malaria parasites in the blood samples.
For the detection of Malaria, it is important to detect infected blood cells. In this approach, I used deep neural networks to detect the malaria in human blood cells. I have shown detection using deep learning technique (CNN). The proposed method in this research shows a system with end-to-end automated models using a deep neural network that performs both feature extraction and classification using blood cell images. The dataset used in this research was taken from the National Institute of Health (NIH) Malaria Dataset. Models are evaluated based on accuracy, precision, recall, and F1score.

For my project, I conducted four experiments, a model employing batch normalization, a second model without batch normalization, and a third experiment in which the model was expanded with adding more layers. Despite, the last experiment without using dropout layer the model’s performance is not optimal as it is overfitting (training accuracy was higher than testing accuracy). According to the results the second model is finalized as it was giving a better output.


## Necessary modules to be imported for the model:
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.model_selection import train_test_split
import seaborn as sns


## I have conducted 4 experiments for this project these including:
1. Model with using batch normalization
2. Model without using batch normalization 
3. Model with adding(2 Convolution with 1 maxpooling) extra layers
4. Model without using dropout layers

All the files in this repository are named with their experiment number, to access these files one can download as .ipynb file to the local machine and can be used. 

Adding to these files, I have also added the saved model wchich ends with the extension of (.h5). So one can download this saved model and use it.
