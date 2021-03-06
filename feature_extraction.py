import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar100

"""
Creates the feature extraction model using VGG-16 based architecture modefied to work on 32x32 images
"""
def create_feats_model(model_name="vgg"):
    if model_name == "vgg":
        # VGG-16 model created using pieces of https://github.com/geifmany/cifar-vgg
        # I've uploaded our assembled model with weights applied as "vgg_model.h5"
        model_w_head = tf.keras.models.load_model("vgg_model.h5")

        # The last feature layer (max pooling) is the 7th from last layer in the full model
        #   (If we don't want pooling before the feature output we can change this index to be -8)
        i_last_feats = -7

        # Removes the classification head so that the model outputs features instead of labels
        output = model_w_head.layers[i_last_feats].output

        # Creates our feature model
        model_feats = tf.keras.Model(inputs=model_w_head.input, outputs=output)

        # Freezes the layers in the model so that they're not trained with the LSTM
        freeze_layers(model_feats)

        print("VGG MODEL")

        return model_feats
    """
    Once the feature extraction CNN is created, we can combine it with the LSTM to create our full model
    full_model = tf.keras.Model(model_feats.input, lstm_model(model_feats.output))
    """

"""
Freezes the layers in the feature extraction model so that its
parameters are not trainable, that way when we combine it with the LSTM only the LSTM will get trained.
"""
def freeze_layers(model):
    for layer in model.layers:
        layer.trainable = False

"""
Normalizes our data based on the CIFAR 100 specs, improves CNN results
(Expects image values to be from 0 to 255)

imgs - the entire set of train or test images (can normalize them all at once before batching)
"""
def normalize_imgs(imgs):

    mean = 121.936 # mean of the CIFAR 100 data
    std = 68.389 # standard deviation of the CIFAR 100 data

    # Adds a small number to the denominators so we don't divide by zero
    norm_imgs = (imgs - mean) / (std + 1e-7)

    return norm_imgs
