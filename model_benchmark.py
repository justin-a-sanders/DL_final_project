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

        self.example_batch_size = 1
        self.batch_size = 16
        self.loss_list = []
        self.acc_list = []

        self.feats_model = create_feats_model("vgg") #Import VGG


    def call(self, inputs, examples):
        """
        Runs a forward pass on an input batch of images examples and test images
        Simply averages the embeddings found by VGG
        """
        examples = tf.squeeze(examples, [0])
        inputs = tf.squeeze(inputs, [0])

        examples = self.feats_model(examples)
        examples = tf.math.reduce_mean(examples, 0)
        inputs = self.feats_model(inputs)

        merged_examples = tf.stack([examples] * 6, axis=0)
        merged_examples = tf.squeeze(merged_examples)
        inputs = tf.squeeze(inputs)

        dist = (-tf.keras.losses.cosine_similarity(merged_examples, inputs) + 1) / 2
        return dist


    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels
        """
        labels = tf.squeeze(tf.cast(labels, tf.float32))
        logits = tf.squeeze(tf.cast(tf.math.greater_equal(logits, 0.6), tf.float32))
        correct_predictions = tf.equal(labels, logits)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def test(model, examples):
    """
    Tests the model on the test inputs and labels.
    """
    acc_list = []
    loss_list = []
    for _ in range(100):
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
            acc = model.accuracy(logits, batch_labels)
            acc_list.append(acc)
            loss_list.append(model.loss(logits, batch_labels))
    return (sum(acc_list)/(len(acc_list))).numpy() ,  (sum(loss_list)/(len(loss_list))).numpy()


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

    examples_train, examples_test = preprocess()
    model = Model(100, 0)

    #Tests the benchmark for different values of the num_examples parameter in the model
    test_accuracies = []
    for num_examples in range(15):
        model.num_examples = model.num_examples + 1
        test_acc, test_loss = test(model, examples_test)
        print(test_acc)
        test_accuracies.append(test_acc)

    x = [i+1 for i in range(len(test_accuracies))]
    plt.plot(x, test_accuracies)
    plt.legend(['Benchmark'])
    plt.title('Accuracy by number of examples')
    plt.xlabel('Number of examples')
    plt.ylabel('Accuracy')
    plt.savefig("benchmark_acc_train.jpg")
    plt.clf()


if __name__ == '__main__':
    main()
