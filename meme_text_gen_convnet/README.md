This repository was created to accompany the article
Meme Text Generation with a Deep Convolutional Network in Keras & Tensorflow

# Contents
 * train.py: script to train the model
 * predict.py: script to make predictions based on a saved model
 * util.py: some custom utility functions for training and predicting
 * training_data_sample.json: a small sample of 10 captions per meme for the top 48 most popular memes,
 taken from public imgflip memes with a minimum number of views. Beware: this is raw data, it contains vulgarity.

# Installation
To run the training and prediction scripts you'll probably want to use a server with a gpu if you want to train quickly,
otherwise any linux box is probably fine for testing out the code. It may or may not work with Mac and Windows, good luck!

## Ubuntu
You'll want to install keras, tensorflow, and h5py in a virtualenv so that the installation doesn't interfere with
other installations on your system. You can skip the first two steps if you already have pip and virtualenv installed.
```bash
sudo apt-get install python3-pip
sudo pip install virtualenv
mkdir my_cool_ml_folder
cd my_cool_ml_folder
virtualenv venv
. venv/bin/activate
pip install keras
pip install tensorflow
pip install h5py

# now you can run the training or prediction script while the venv is activated
```
