# Font-Recognition
The model's accuracy might be compromised when handling unfamiliar data, particularly if the data differs significantly from the examples it was trained on. For optimal performance, it's advisable to use images that closely resemble those used during the model's training phase.

This project is focused on building a font detection system employing Convolutional Neural Networks (CNNs). The system is designed to analyze images containing text composed in various fonts and then predict the specific font used.

Installation
To execute the code provided in this repository, ensure you have the following dependencies installed:

Python 3.x
OpenCV (cv2)
NumPy
TensorFlow
Pandas
Matplotlib
scikit-learn
You can easily install these required Python packages using the command pip install opencv-python numpy tensorflow pandas matplotlib scikit-learn.

Dataset
The dataset comprises images containing text samples written in diverse fonts. Each image is tagged with the name of the font used. The dataset is structured into subdirectories, with each subdirectory representing a distinct font.

Data Loading and Preprocessing
The process involves loading images from the dataset directory, resizing them to a predefined target size, converting them to grayscale, and then normalizing them.

The dataset is divided into three subsets: training, validation, and test sets.

Model Architecture
The font detection model is constructed using a CNN architecture. It comprises four convolutional layers, each followed by max-pooling layers. Additionally, it includes a flatten layer, a fully connected layer, dropout regularization, and an output layer.

Training
Training of the model occurs using the training set, with validation performed on the validation set. Early stopping criteria are applied during training, halting the process if the validation loss fails to improve after a specified number of epochs.

Evaluation
Post-training, the model's performance in font detection is evaluated using the test set. Metrics such as test loss and accuracy are computed and reported.
