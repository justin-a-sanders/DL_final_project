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

		self.batch_size = 100
		self.loss_list = []
		self.acc_list = []

		self.learning_rate = 0.001
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		self.normalize_1 = tf.keras.layers.BatchNormalization()
		self.normalize_2 = tf.keras.layers.BatchNormalization()
		self.normalize_3 = tf.keras.layers.BatchNormalization()


		self.conv_1 = tf.keras.layers.Conv2D(16, 5, strides = (2,2), padding='SAME', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
		self.pool_1 = tf.keras.layers.MaxPool2D((3, 3), strides=(2,2))
		self.conv_2 = tf.keras.layers.Conv2D(32, 3, strides = (2,2), padding='SAME', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
		self.pool_2 = tf.keras.layers.MaxPool2D((2, 2), strides=(2,2))
		self.conv_3 = tf.keras.layers.Conv2D(32, 3, strides = (1,1), padding='SAME', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.random_normal_initializer(stddev=0.1))
		self.lstm = tf.keras.layers.LSTM(32, return_sequences=True, return_state=True)
		self.distance = tf.keras.layers.Dense(2)

	def call(self, inputs, examples):
		"""
		Runs a forward pass on an input batch of images examples and test images
		"""
		examples = self.conv_1(examples)
		examples = self.normalize_1(examples)
		examples = self.pool_1(examples)
		examples = self.conv_2(examples)
		examples = self.normalize_2(examples)
		examples = self.pool_2(examples)
		examples = self.conv_3(examples)
		examples = self.normalize_3(examples)
		examples = tf.reshape(examples, [self.num_examples, -1])
		_, merged_examples, _ = self.lstm(tf.expand_dims(examples, axis=0), initial_state=None)

		inputs = self.conv_1(inputs)
		inputs = self.normalize_1(inputs)
		inputs = self.pool_1(inputs)
		inputs = self.conv_2(inputs)
		inputs = self.normalize_2(inputs)
		inputs = self.pool_2(inputs)
		inputs = self.conv_3(inputs)
		inputs = self.normalize_3(inputs)
		inputs = tf.reshape(inputs, [self.batch_size, -1])

		merged_examples = tf.stack([merged_examples[0]] * self.batch_size)

		dist = (-tf.keras.losses.cosine_similarity(merged_examples, inputs) + 1) / 2

		#dist = -self.similarity(merged_examples, inputs)

		#print(dist.shape)
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
		logits = tf.cast(tf.math.greater_equal(logits, 0.5), tf.float32)
		correct_predictions = tf.equal(tf.cast(labels, tf.float32), logits)
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, examples):
	'''
	Trains the model on all of the inputs and labels for one epoch.
	'''
	#indices = tf.random.shuffle(range(0, len(train_labels)))
	#train_inputs = tf.gather(train_inputs, indices)
	#train_inputs = tf.image.random_flip_left_right(train_inputs)
	#train_labels = tf.gather(train_labels, indices)
	for ii in range(len(examples)):
		example_indices = np.random.choice(len(examples[ii]), model.num_examples)
		batch_examples = examples[ii][example_indices]

		batch_pos_indices = np.random.choice(len(examples[ii]), int(model.batch_size/2))
		batch_pos_inputs = examples[ii][batch_pos_indices]
		batch_pos_labels = np.ones((int(model.batch_size/2)))

		batch_neg_indices = (np.random.choice(len(examples), int(model.batch_size/2)) , np.random.choice(len(examples[ii]), int(model.batch_size/2)))
		batch_neg_inputs = examples[batch_neg_indices]
		batch_neg_labels = np.zeros((int(model.batch_size/2)))

		batch_inputs = np.concatenate((batch_pos_inputs, batch_neg_inputs))
		batch_labels = np.concatenate((batch_pos_labels, batch_neg_labels))
		# print(batch_examples.shape, "train_examples")
		# print(batch_inputs.shape, "train_inputs")

		with tf.GradientTape() as tape:
			logits = model.call(batch_inputs, batch_examples)
			loss = model.loss(logits, batch_labels)
			model.loss_list.append(loss.numpy())

			acc = model.accuracy(logits, batch_labels)
			model.acc_list.append(acc.numpy())

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, negative_examples):
	"""
	Tests the model on the test inputs and labels.
	"""
	negative_examples = np.reshape(negative_examples, (negative_examples.shape[0]*negative_examples.shape[1], negative_examples.shape[2], negative_examples.shape[3], negative_examples.shape[4]))
	total_accuracy = []
	for label in range(test_inputs.shape[0]):
		np.random.shuffle(negative_examples.T)
		np.random.shuffle(test_inputs.T)
		batch_inputs = test_inputs[label][int(model.batch_size/2):int(model.batch_size/2)+model.num_examples]
		batch_neg_examples = negative_examples[:int(model.batch_size/2)]
		batch_pos_examples = test_inputs[label][:int(model.batch_size/2)]
		examples = np.concatenate((batch_pos_examples, batch_neg_examples))
		labels = np.concatenate((np.ones(int(model.batch_size/2)), np.zeros(int(model.batch_size/2))))
		# print(batch_inputs.shape, "test_inputs")
		# print(examples.shape, "examples_test")
		total_accuracy.append(model.accuracy(model.call(examples, batch_inputs), labels))
	return np.mean(total_accuracy)
	


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
	plt.clf()

def visualize_acc(accuracy_list):
	"""
	Uses Matplotlib to visualize the losses of our model.
	:param losses: list of loss data stored from train. Can use the model's loss_list
	field
	"""
	x = [i for i in range(len(accuracy_list))]
	plt.plot(x, accuracy_list)
	plt.title('Train Accuracy per batch')
	plt.xlabel('Batch')
	plt.ylabel('Acc')
	plt.savefig("episode_acc.jpg")
	plt.clf()

def visualize_test_acc(accuracy_list):
	"""
	Uses Matplotlib to visualize the losses of our model.
	:param losses: list of loss data stored from train. Can use the model's loss_list
	field
	"""
	x = [i for i in range(len(accuracy_list))]
	plt.plot(x, accuracy_list)
	plt.title('Test Accuracy per batch')
	plt.xlabel('Batch')
	plt.ylabel('Acc')
	plt.savefig("test_acc.jpg")
	plt.clf()


def preprocess():
	#Load in the CIFAR 100 dataset
	# (train_data1, train_labels1), (train_data2, train_labels2) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
	(train_data, train_labels), (test_data, _) = tf.keras.datasets.mnist.load_data()

	# train_data = [i for i in train_data1]
	# train_data += [j for j in train_data2]
	# train_data = np.asarray(train_data)
	# train_labels = np.append(train_labels1, train_labels2)


	examples = [[] for ii in range(10)]
	test_examples =  [[] for ii in range(10)]
	for ii in range(len(train_labels)):
		examples[train_labels[ii]].append(train_data[ii])
	for j in range(len(test_data)):
		test_examples[train_labels[j]].append(test_data[j])

	for i in range(len(examples)):
		examples[i] = examples[i][:5000]
		test_examples[i] = test_examples[i][:800]
		# print(len(test_examples[i]))

	examples = np.expand_dims(np.asarray(examples).astype(np.float32)/255, axis=-1)
	test_examples = np.expand_dims(np.asarray(test_examples).astype(np.float32)/255, axis=-1)
	return examples, test_examples


def main():
	#get train data
	examples, test_negative_examples = preprocess()
	train_data = examples[:8]
	test_data = examples[8:]
	test_negative_examples = test_negative_examples[:8]
	model = Model(100, 5)

	test_acc = []
	for epoch in range(500):
		train(model, train_data)
		test_acc.append(test(model, test_data, test_negative_examples))
		if epoch % 10 == 0:
			visualize_loss(model.loss_list)
			visualize_acc(model.acc_list)
			visualize_test_acc(test_acc)
		print(epoch)


if __name__ == '__main__':
	main()
