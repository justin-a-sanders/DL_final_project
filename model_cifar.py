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

        #self.normalize1 = tf.keras.layers.BatchNormalization()

        self.conv_3 = tf.keras.layers.Conv2D(64, 3, strides = (1,1), padding='SAME', activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        self.conv_4 = tf.keras.layers.Conv2D(64, 3, strides = (1,1), activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        #self.embed = tf.keras.layers.Dense(128, activation='elu', kernel_initializer=tf.random_normal_initializer(stddev=0.1))

        #self.normalize2 = tf.keras.layers.BatchNormalization()

        #self.conv_5 = tf.keras.layers.Conv2D(256, 3, strides = (1,1), padding='SAME', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        #self.conv_6 = tf.keras.layers.Conv2D(256, 3, strides = (1,1), activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        #self.pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        #self.normalize3 = tf.keras.layers.BatchNormalization()

        self.lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)

        #self.distance = tf.keras.layers.Dense(2)

    def call(self, inputs, examples):
        """
        Runs a forward pass on an input batch of images examples and test images
        """
        examples = tf.reshape(examples, (self.example_batch_size * self.num_examples, 32, 32, 3))
        examples = self.conv_1(examples)
        examples = self.conv_2(examples)
        #examples = self.normalize1(examples)
        examples = self.pool_1(examples)

        examples = self.conv_3(examples)
        examples = self.conv_4(examples)
        #examples = self.normalize2(examples)
        examples = self.pool_2(examples)

        #examples = self.conv_5(examples)
        #examples = self.conv_6(examples)
        #examples = self.normalize3(examples)
        #examples = self.pool_3(examples)
        #examples = tf.reshape(examples, (self.example_batch_size * self.num_examples, -1))
        #examples = self.embed(examples)

        examples = tf.reshape(examples, (self.example_batch_size, self.num_examples, -1))

        _, merged_examples, _ = self.lstm(examples, initial_state=None)

        #merged_examples = tf.reshape(examples, (self.example_batch_size, -1))

        inputs = tf.reshape(inputs, (self.example_batch_size * self.batch_size, 32, 32, 3))
        inputs = self.conv_1(inputs)
        inputs = self.conv_2(inputs)
        #inputs = self.normalize1(inputs)
        inputs = self.pool_1(inputs)

        inputs = self.conv_3(inputs)
        inputs = self.conv_4(inputs)
        #inputs = self.normalize2(examples)
        inputs = self.pool_2(inputs)

        #inputs = self.conv_5(inputs)
        #inputs = self.conv_6(inputs)
        #inputs = self.normalize3(examples)
        #inputs = self.pool_3(inputs)
        #inputs = tf.reshape(inputs, (self.example_batch_size * self.batch_size, -1))
        #inputs = self.embed(inputs)

        inputs = tf.reshape(inputs, (self.example_batch_size, self.batch_size, -1))

        test = tf.stack([merged_examples] * self.example_batch_size, axis=1)
        ex_dist = (-tf.keras.losses.cosine_similarity(test, examples) + 1) / 2
        print(ex_dist.numpy())

        merged_examples = tf.stack([merged_examples] * self.batch_size, axis=1)
        dist = (-tf.keras.losses.cosine_similarity(merged_examples, inputs) + 1) / 2
        return dist

        #combined = tf.concat([merged_examples, inputs], 1)
        #logits = self.distance(combined)
        #return logits



    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        """
        #return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))
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
    #indices = tf.random.shuffle(range(0, len(train_labels)))
    #train_inputs = tf.gather(train_inputs, indices)
    #train_inputs = tf.image.random_flip_left_right(train_inputs)
    #train_labels = tf.gather(train_labels, indices)
    accuracies = []
    losses = []
    indices = tf.random.shuffle(range(0, len(examples)))
    examples = examples[indices]
    #examples = np.asarray([tf.image.random_flip_left_right(i) for i in examples])
    for i in range(len(examples)):
        examples[i] = tf.image.random_flip_left_right(examples[i])

    for ii in range(0, len(examples), model.example_batch_size):
        example_indices = np.random.choice(len(examples[ii]), model.num_examples)
        batch_examples = examples[ii:ii+model.example_batch_size,example_indices]
        #print(batch_examples.shape)

        batch_pos_indices = np.random.choice(len(examples[ii]), int(model.batch_size/2))
        batch_pos_inputs = examples[ii:ii+model.example_batch_size,batch_pos_indices]
        batch_pos_labels = np.zeros((model.example_batch_size, int(model.batch_size/2)))
        batch_pos_labels += 1
        #print(batch_pos_inputs.shape)

        batch_neg_indices = (np.random.choice(len(examples), int(model.batch_size/2)) , np.random.choice(len(examples[ii]), int(model.batch_size/2)))
        batch_neg_inputs = np.asarray([examples[batch_neg_indices] for i in range(model.example_batch_size)])
        batch_neg_labels = np.zeros((model.example_batch_size, int(model.batch_size/2)))
        #print(batch_neg_inputs.shape)
        #batch_neg_labels[:,0] = 1

        batch_inputs = np.concatenate((batch_pos_inputs, batch_neg_inputs), axis=1)
        #print(batch_inputs.shape)
        batch_labels = np.concatenate((batch_pos_labels, batch_neg_labels), axis=1)
        #print(batch_labels.shape)

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
    train_labels = np.append(train_labels1, train_labels2)

    examples = [[] for ii in range(100)]
    for ii in range(len(train_labels)):
        examples[train_labels[ii]].append(train_data[ii])

    examples_train = np.asarray(examples).astype(np.float32)/255


    (_, _), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()
    examples = [[] for ii in range(10)]
    for ii in range(len(test_labels)):
        examples[test_labels[ii][0]].append(test_data[ii])

    examples_test = np.asarray(examples).astype(np.float32)/255

    return examples_train, examples_test


def main():
    #get train data
    examples_train, examples_test = preprocess()
    print(examples_train.shape)
    print(examples_test.shape)

    model = Model(100, 5)

    losses = []
    accuracies = []
    test_accuracies = []
    test_losses = []
    for epoch in range(10000):
        print(epoch)
        visualize_loss(losses, test_losses)
        visualize_acc(accuracies, test_accuracies)

        loss, acc = train(model, examples_train)
        losses.append(loss)
        accuracies.append(acc)

        test_acc, test_loss = test(model, examples_test)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        print(test_acc)


if __name__ == '__main__':
    main()
