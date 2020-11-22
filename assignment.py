from __future__ import absolute_import
from matplotlib import pyplot as plt


import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self, num_classes, num_examples):
        """
        Define architechture for the model
        """
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.num_examples = num_examples

        self.batch_size = 100
        self.loss_list = []

        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


    def call(self, inputs, examples):
        """
        Runs a forward pass on an input batch of images examples and test images
        """
        print(inputs.shape)
        print(examples.shape)
        print(" ")
        pass

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        """
        pass

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels
        """
        pass

def train(model, examples, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch.
    '''
    losses = []
    indices = tf.random.shuffle(range(0, len(train_labels)))
    train_inputs = tf.gather(train_inputs, indices)
    train_inputs = tf.image.random_flip_left_right(train_inputs)
    train_labels = tf.gather(train_labels, indices)

    for ii in range(0, len(train_labels), model.batch_size):
        batch_inputs = train_inputs[ii : ii + model.batch_size]
        batch_labels = train_labels[ii : ii + model.batch_size]

        example_label = random.randint(0,model.num_classes-1)
        example_indices = np.random.choice(len(examples[example_label]), model.num_examples)
        batch_examples = examples[example_label][example_indices]

        print(batch_labels.shape)
        model.call(batch_inputs, batch_examples)

        #with tf.GradientTape() as tape:
        #    predictions =
        #    loss =
        #    model.loss_list.append(loss)

        #gradients = tape.gradient(loss, model.trainable_variables)
        #optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.
    """



def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig("episode_loss.jpg")


def preprocess():
    #Load in the CIFAR 100 dataset
    (train_data1, train_labels1), (train_data2, train_labels2) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    train_data = [i for i in train_data1]
    train_data += [j for j in train_data2]
    train_data = np.asarray(train_data)
    train_labels = np.append(train_labels1, train_labels2)

    examples = [[] for ii in range(100)]
    for ii in range(len(train_labels)):
        examples[train_labels[ii]].append(train_data[ii])

    examples = np.asarray(examples)
    return examples, train_data, train_labels


def main():
    #get train data
    examples, train_data, train_labels = preprocess()
    print(examples.shape)
    print(train_data.shape)
    print(train_labels.shape)

    model = Model(100, 5)

    train(model, examples, train_data, train_labels)


if __name__ == '__main__':
    main()
