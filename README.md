**Image Colorizer**
This project utilizes a deep learning model to colorize black-and-white images. The model is based on a pre-trained Caffe model. The application allows users to upload a grayscale image, process it with the colorization model, and download the colorized image.

**Features**
Upload black and white images.

Colorize the uploaded image using a deep learning model.

Download the colorized image.

## 🧠 Key Deep Learning Technologies Used

- **Convolutional Neural Networks (CNNs)**: Used to extract features from grayscale images and predict appropriate color values.
- **Caffe Framework**: The colorization model is trained and run using Caffe.
- **Pre-trained Caffe Model**: A model trained on a large dataset (ImageNet) for image colorization tasks.
- **LAB Color Space**: Images are converted to LAB color space, where the 'L' channel is the input, and the model predicts 'a' and 'b' channels for color.

**Requirements**
Python 3.x

Flask: A web framework for creating the web app.

OpenCV: For image processing.

NumPy: For numerical operations.

Caffe Model: Pre-trained model for colorization.

**Download the necessary model files:**
Caffe Model: You can download the pre-trained Caffe model files:

colorization_deploy_v2.prototxt

colorization_release_v2.caffemodel

pts_in_hull.npy


