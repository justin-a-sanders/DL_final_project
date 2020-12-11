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
    def __init__(self, num_classes, num_examples, vgg):
        """
        Define architecture for the model
        """
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.num_examples = num_examples
        self.vgg = vgg

        self.similarity = tf.keras.losses.CosineSimilarity(axis = 1)

        if self.vgg:
            self.kqv_size = 512 #these sizes must match the output size of VGG
            self.embedding_size = 512 
        else:
            self.kqv_size = 256
            self.embedding_size = 256

        #initializing transformer weights
        self.W_k = tf.Variable(tf.random.truncated_normal([self.embedding_size, self.kqv_size], mean=0.0, stddev=.1))
        self.W_q = tf.Variable(tf.random.truncated_normal([self.embedding_size, self.kqv_size], mean=0.0, stddev=.1))
        self.W_v = tf.Variable(tf.random.truncated_normal([self.embedding_size, self.kqv_size], mean=0.0, stddev=.1))
        
        self.example_batch_size = 5
        self.batch_size = 64
        self.loss_list = []
        self.acc_list = []

        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if self.vgg:
            self.feats_model = create_feats_model("vgg")
        else:
            #if not using VGG, use this 6 layer CNN
            self.conv_1 = tf.keras.layers.Conv2D(32, 3, strides = (2,2), padding='SAME', activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            self.conv_2 = tf.keras.layers.Conv2D(32, 3, strides = (1,1), activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

            self.conv_3 = tf.keras.layers.Conv2D(64, 3, strides = (1,1), padding='SAME', activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            self.conv_4 = tf.keras.layers.Conv2D(64, 3, strides = (1,1), activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))


    def call(self, inputs, examples):
        """
        Runs a forward pass on an input batch of images examples and test images
        """
        examples = tf.reshape(examples, (self.example_batch_size * self.num_examples, 32, 32, 3))

        if self.vgg:
            examples = self.feats_model(examples)
        else:
            examples = self.conv_1(examples)
            examples = self.conv_2(examples)
            examples = self.pool_1(examples)

            examples = self.conv_3(examples)
            examples = self.conv_4(examples)
            examples = self.pool_2(examples)

        examples = tf.reshape(examples, (self.example_batch_size, self.num_examples, -1))

        Z = []
        dk = tf.sqrt(tf.cast(self.kqv_size, tf.float32))

        #loops to avoid issues with high order matrix multiplication
        for i in range(self.example_batch_size):
            K = tf.matmul(examples[i], self.W_k)
            Q = tf.matmul(examples[i], self.W_q)
            V = tf.matmul(examples[i], self.W_v)

            Z.append(tf.matmul(tf.nn.softmax(tf.matmul(Q,tf.transpose(K))/dk), V))

        merged_examples = tf.reduce_mean(tf.convert_to_tensor(Z),axis=1)

        inputs = tf.reshape(inputs, (self.example_batch_size * self.batch_size, 32, 32, 3))

        if self.vgg:
            inputs = self.feats_model(inputs)
        else:
            inputs = self.conv_1(inputs)
            inputs = self.conv_2(inputs)
            inputs = self.pool_1(inputs)

            inputs = self.conv_3(inputs)
            inputs = self.conv_4(inputs)
            inputs = self.pool_2(inputs)

        inputs = tf.reshape(inputs, (self.example_batch_size, self.batch_size, -1))

        merged_examples = tf.stack([merged_examples] * self.batch_size, axis=1)

        dist = (-tf.keras.losses.cosine_similarity(merged_examples, inputs) + 1) / 2

        return dist

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        """
        return tf.reduce_sum(tf.sqrt(tf.abs((logits - tf.cast(labels, tf.float32)))))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels
        """
        labels = tf.cast(labels, tf.float32)
        logits = tf.cast(tf.math.greater_equal(logits, 0.6), tf.float32)
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


def visualize_loss(model, train_loses, test_loses):
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

    if model.vgg:
        filename = "TRANS_VGG_LOSS_lr" + str(model.learning_rate) + "ex_batch_sz" + str(model.example_batch_size) + ".jpg"
    else:
        filename = "TRANS_NOVGG_LOSS_lr" + str(model.learning_rate) + "ex_batch_sz" + str(model.example_batch_size) + ".jpg"
    plt.savefig(filename)
    plt.clf()

def visualize_acc(model, train_acc, test_acc):
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

    if model.vgg:
        filename = "TRANS_VGG_ACC_lr" + str(model.learning_rate) + "ex_batch_sz" + str(model.example_batch_size) + ".jpg"
    else:
        filename = "TRANS_NOVGG_ACC_lr" + str(model.learning_rate) + "ex_batch_sz" + str(model.example_batch_size) + ".jpg"
    plt.savefig(filename)
    plt.clf()


def preprocess():
    #Load in the CIFAR 100 dataset
    (train_data1, train_labels1), (train_data2, train_labels2) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    train_data = [i for i in train_data1]
    train_data += [j for j in train_data2]
    train_data = np.asarray(train_data)
    train_data = normalize_imgs(train_data)  # normalizes our data based on the CIFAR 100 specs, improves CNN results
    train_labels = np.append(train_labels1, train_labels2)

    examples = [[] for ii in range(100)]
    for ii in range(len(train_labels)):
        examples[train_labels[ii]].append(train_data[ii])

    examples_train = np.asarray(examples).astype(np.float32)

    (_, _), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()
    test_data = normalize_imgs(test_data)  # normalizes the test images
    examples = [[] for ii in range(10)]
    for ii in range(len(test_labels)):
        examples[test_labels[ii][0]].append(test_data[ii])

    examples_test = np.asarray(examples).astype(np.float32)

    return examples_train, examples_test


def main():
    # Changing this boolean chooses whether to use pretrained feature extraction or not
    vgg=True

    #get train data
    examples_train, examples_test = preprocess()
    # print(examples_train.shape)
    # print(examples_test.shape)

    model = Model(100, 5, vgg=vgg)

    losses = []
    accuracies = []
    test_accuracies = []
    test_losses = []
    for epoch in range(1300):
        print(epoch)
        visualize_loss(model, losses, test_losses)
        visualize_acc(model, accuracies, test_accuracies)

        loss, acc = train(model, examples_train)
        losses.append(loss)
        accuracies.append(acc)

        test_acc, test_loss = test(model, examples_test)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

        # print("Time for epoch:", (datetime.now() - start))
        print("Epoch:", epoch)
        print("Test Acc:", test_acc)
        if epoch % 10 == 0:
            visualize_loss(model, losses, test_losses)
            visualize_acc(model, accuracies, test_accuracies)


if __name__ == '__main__':
    main()
