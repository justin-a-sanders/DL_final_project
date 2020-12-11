from __future__ import absolute_import
from matplotlib import pyplot as plt


import os
import tensorflow as tf
import numpy as np
import random
import math
from datetime import datetime
from feature_extraction import create_feats_model, normalize_imgs

class Model(tf.keras.Model):
    def __init__(self, num_classes, num_examples):
        """
        Define architechture for the model
        """
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.num_examples = num_examples

        self.similarity = tf.keras.losses.CosineSimilarity(axis = 1)

        self.example_batch_size = 5
        self.batch_size = 64
        self.loss_list = []
        self.acc_list = []

        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.conv_1 = tf.keras.layers.Conv2D(32, 3, strides = (2,2), padding='SAME', activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        self.conv_2 = tf.keras.layers.Conv2D(32, 3, strides = (1,1), activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.normalize1 = tf.keras.layers.BatchNormalization()

        self.conv_3 = tf.keras.layers.Conv2D(64, 3, strides = (1,1), padding='SAME', activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        self.conv_4 = tf.keras.layers.Conv2D(64, 3, strides = (1,1), activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.embed = tf.keras.layers.Dense(128, activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.1))

        self.normalize2 = tf.keras.layers.BatchNormalization()

        self.conv_5 = tf.keras.layers.Conv2D(256, 3, strides = (1,1), padding='SAME', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        self.conv_6 = tf.keras.layers.Conv2D(256, 3, strides = (1,1), activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        self.pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.normalize3 = tf.keras.layers.BatchNormalization()

        self.feats_model = create_feats_model("vgg")

        self.input_dense = tf.keras.layers.Dense(512)

        self.gru = tf.keras.layers.GRU(512, return_sequences=True, return_state=True)

    def call(self, inputs, examples):
        """
        Runs a forward pass on an input batch of images examples and test images
        """

        examples = tf.reshape(examples, (self.example_batch_size * self.num_examples, 32, 32, 3))

        # if using handmade CNN
        # examples = self.conv_1(examples)
        # examples = self.conv_2(examples)
        # examples = self.pool_1(examples)

        # examples = self.conv_3(examples)
        # examples = self.conv_4(examples)
        # examples = self.pool_2(examples)

        #if using VGG
        examples = self.feats_model(examples)

        examples = tf.reshape(examples, (self.example_batch_size, self.num_examples, -1))


        #combining grus of examples in two oposite orders
        _, merged_examples = self.gru(examples, initial_state=None)
        _, merged_examples_2 = self.gru(tf.reverse(examples, [1]), initial_state=None)

        #take the mean of the two grus
        merged_examples = tf.reduce_mean(tf.stack([merged_examples, merged_examples_2]), axis=0)


        inputs = tf.reshape(inputs, (self.example_batch_size * self.batch_size, 32, 32, 3))

        # if using handmade CNN
        # inputs = self.conv_1(inputs)
        # inputs = self.conv_2(inputs)
        # inputs = self.pool_1(inputs)

        # inputs = self.conv_3(inputs)
        # inputs = self.conv_4(inputs)
        # inputs = self.pool_2(inputs)

        #if using VGG
        inputs = self.feats_model(inputs)

        inputs = tf.reshape(inputs, (self.example_batch_size, self.batch_size, -1))

        inputs = self.input_dense(inputs)

        merged_examples = tf.stack([merged_examples] * self.batch_size, axis=1)
        dist = (-tf.keras.losses.cosine_similarity(merged_examples, inputs) + 1) / 2
        return dist


    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        """
        return tf.reduce_sum(tf.square(logits - tf.cast(labels, tf.float32)))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels
        """
        labels = tf.cast(labels, tf.float32)
        logits = tf.cast(tf.math.greater_equal(logits, 0.5), tf.float32)
        correct_predictions = tf.equal(labels, logits)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, examples):
    '''
    Trains the model on all of the inputs and labels for one epoch.
    '''
    accuracies = []
    losses = []
    indices = tf.random.shuffle(range(0, len(examples)))
    examples = examples[indices]

    for i in range(len(examples)):
        examples[i] = tf.image.random_flip_left_right(examples[i])

    for ii in range(0, len(examples), model.example_batch_size):
        example_indices = np.random.choice(len(examples[ii]), model.num_examples)
        batch_examples = examples[ii:ii+model.example_batch_size,example_indices]

        batch_pos_indices = np.random.choice(len(examples[ii]), int(model.batch_size/2))
        batch_pos_inputs = examples[ii:ii+model.example_batch_size,batch_pos_indices]
        batch_pos_labels = np.zeros((model.example_batch_size, int(model.batch_size/2)))
        batch_pos_labels += 1

        batch_neg_indices = (np.random.choice(len(examples), int(model.batch_size/2)) , np.random.choice(len(examples[ii]), int(model.batch_size/2)))
        batch_neg_inputs = np.asarray([examples[batch_neg_indices] for i in range(model.example_batch_size)])
        batch_neg_labels = np.zeros((model.example_batch_size, int(model.batch_size/2)))

        batch_inputs = np.concatenate((batch_pos_inputs, batch_neg_inputs), axis=1)
        batch_labels = np.concatenate((batch_pos_labels, batch_neg_labels), axis=1)

        with tf.GradientTape() as tape:
            logits = model.call(batch_inputs, batch_examples)
            loss = model.loss(logits, batch_labels)
            losses.append(loss.numpy())

            acc = model.accuracy(logits, batch_labels)
            accuracies.append(acc.numpy())

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return sum(losses)/len(losses) , sum(accuracies)/len(accuracies)

def test(model, examples):
    """
    Tests the model on the test inputs and labels.
    """
    acc_list = []
    loss_list = []
    for _ in range(5):
        for ii in range(0, len(examples), model.example_batch_size):
            example_indices = np.random.choice(len(examples[ii]), model.num_examples)
            batch_examples = examples[ii:ii+model.example_batch_size,example_indices]

            batch_pos_indices = np.random.choice(len(examples[ii]), int(model.batch_size/2))
            batch_pos_inputs = examples[ii:ii+model.example_batch_size,batch_pos_indices]
            batch_pos_labels = np.zeros((model.example_batch_size, int(model.batch_size/2)))
            batch_pos_labels += 1

            batch_neg_indices = (np.random.choice(len(examples), int(model.batch_size/2)) , np.random.choice(len(examples[ii]), int(model.batch_size/2)))
            batch_neg_inputs = np.asarray([examples[batch_neg_indices] for i in range(model.example_batch_size)])
            batch_neg_labels = np.zeros((model.example_batch_size, int(model.batch_size/2)))

            batch_inputs = np.concatenate((batch_pos_inputs, batch_neg_inputs), axis=1)
            batch_labels = np.concatenate((batch_pos_labels, batch_neg_labels), axis=1)

            logits = model.call(batch_inputs, batch_examples)
            acc_list.append(model.accuracy(logits, batch_labels))
            loss_list.append(model.loss(logits, batch_labels))
    return (sum(acc_list)/(len(acc_list))).numpy() ,  (sum(loss_list)/(len(loss_list))).numpy()


def visualize_loss(train_loses, test_loses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    """
    x = [i for i in range(len(train_loses))]
    plt.plot(x, train_loses)
    plt.plot(x, test_loses)
    plt.legend(['Train', 'Test'])
    plt.title('Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("episode_loss.jpg")
    plt.clf()

def visualize_acc(train_acc, test_acc):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    """
    x = [i for i in range(len(train_acc))]
    plt.plot(x, train_acc)
    plt.plot(x, test_acc)
    plt.legend(['Train', 'Test'])
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("episode_acc.jpg")
    plt.clf()


def preprocess():
    #Load in the CIFAR 100 dataset
    (train_data1, train_labels1), (train_data2, train_labels2) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    train_data = [i for i in train_data1]
    train_data += [j for j in train_data2]
    train_data = np.asarray(train_data)
    train_data = normalize_imgs(train_data) # normalizes our data based on the CIFAR 100 specs, improves CNN results
    train_labels = np.append(train_labels1, train_labels2)

    examples = [[] for ii in range(100)]
    for ii in range(len(train_labels)):
        examples[train_labels[ii]].append(train_data[ii])

    examples_train = np.asarray(examples).astype(np.float32)


    (_, _), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()
    test_data = normalize_imgs(test_data) # normalizes the test images
    examples = [[] for ii in range(10)]
    for ii in range(len(test_labels)):
        examples[test_labels[ii][0]].append(test_data[ii])

    examples_test = np.asarray(examples).astype(np.float32)

    return examples_train, examples_test


def main():
    #get train data
    examples_train, examples_test = preprocess()

    model = Model(100, 5)

    losses = []
    accuracies = []
    test_accuracies = []
    test_losses = []
    for epoch in range(1500):
        start = datetime.now()
        loss, acc = train(model, examples_train)
        losses.append(loss)
        accuracies.append(acc)

        test_acc, test_loss = test(model, examples_test)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

        print("Time for epoch:", (datetime.now() - start))
        if epoch % 10 == 0:
            print("Epoch", epoch)
            print("Test Acc:", test_acc)

            visualize_loss(losses, test_losses)
            visualize_acc(accuracies, test_accuracies)


if __name__ == '__main__':
    main()
