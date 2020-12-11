import tensorflow as tf
from tensorflow.python.keras.utils import data_utils


# This builds an architecture similar to VGG-19 to create our image embeddings
# Our network removes the additional dense layers to only output feature embeddings instead of classifications
# Pretrained weights from ImageNet are then downloaded and applied to the network so that we don't have to train it

# Builds our VGG-19-esque network
def make_vgg(img_height, img_width):
    img_input = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = tf.keras.Model(img_input, x)

    # Load in weights and apply them to our network
    weights_no_top = ('https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    weights_path = data_utils.get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', weights_no_top)
    model.load_weights(weights_path)

    # Setting the layers as untrainable since we've already loaded in weights
    for layer in model.layers:
        layer.trainable = False

    model.summary()

    return model
