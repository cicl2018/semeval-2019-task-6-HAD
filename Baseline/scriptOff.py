#!/usr/bin/python3
"""
Authors:	Himanshu Bansal, 4232858
			Daniel Nagel,  3098420
			Anita Soloveva, 4265100

Description: 

Dependency: Preinstalled Dataset for ekphrasis 
"""

import sys
import re
import numpy as np
from enum import Enum
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import rnn
from ekphrasis.classes.segmenter import Segmenter
import warnings
warnings.simplefilter("ignore")
# Twitter Hashtag Parser 
tw = Segmenter(corpus="twitter")

# Configuration class for training model.
class Configuration:
	num_epochs = 20
	size_batch = 128
	max_time_steps = 40
	LSTM_CT = 4
	LSTM_SZ = 200
	ratio_dropout = 0.95
	embedding_size = 100
	learning_rate = 0.001

class Phase(Enum):
	Train = 0
	Validation = 1
	Predict = 2

class Model:
	def __init__(self, config, batch, lens_batch, label_batch, n_chars, numberer, phase = Phase.Predict):
		size_batch = batch.shape[1]
		input_size = batch.shape[2]
		label_size = label_batch.shape[2]
		
		# The integer-encoded words. input_size is the (maximum) number of
		# time steps.
		self._x = tf.placeholder(tf.int32, shape=[size_batch, input_size])

		# This tensor provides the actual number of time steps for each
		# instance.
		self._lens = tf.placeholder(tf.int32, shape=[size_batch])

		# The label distribution.
		if phase != Phase.Predict:
			self._y = tf.placeholder(
				tf.float32, shape=[size_batch, label_size])

		# convert to embeddings
		embeddings = tf.get_variable("embeddings", shape = [n_chars, config.embedding_size])
		input_layer = tf.nn.embedding_lookup(embeddings, self._x)

		# make a bunch of LSTM cells and link them
		# use rnn.DropoutWrapper instead of tf.nn.dropout because the layers are anonymous
		stacked_LSTM = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(config.LSTM_SZ), output_keep_prob = config.ratio_dropout) for _ in range(config.LSTM_CT)])
				
		# run the whole thing
		_, hidden = tf.nn.dynamic_rnn(stacked_LSTM, input_layer, sequence_length = self._lens, dtype = tf.float32)
		w = tf.get_variable("W", shape=[hidden[-1].h.shape[1], label_size]) # Acording to the structure of MultiRNNCell, The hidden[-1] is the final state
		b = tf.get_variable("b", shape=[1])
		logits = tf.matmul(hidden[-1].h, w) + b

		if phase == Phase.Train or Phase.Validation:
			losses = tf.nn.softmax_cross_entropy_with_logits(
				labels=self._y, logits=logits)
			self._loss = loss = tf.reduce_sum(losses)

		if phase == Phase.Train:
			self._train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(losses)
			self._probs = probs = tf.nn.softmax(logits)

		if phase == Phase.Validation:
			# Highest probability labels of the gold data.
			gs_labels = tf.argmax(self._y, axis=1)
			# Predicted labels
			self._hp_labels = tf.argmax(logits, axis=1)

			correct = tf.equal(self._hp_labels, gs_labels)
			correct = tf.cast(correct, tf.float32)

			self._accuracy = tf.reduce_mean(correct)
			#self._hp_labels = hp_labels

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def hp_labels(self) :
		return self._hp_labels
	
	@property
	def lens(self):
		return self._lens

	@property
	def loss(self):
		return self._loss

	@property
	def probs(self):
		return self._probs

	@property
	def train_op(self):
		return self._train_op

	@property
	def x(self):
		return self._x

	@property
	def y(self):
		return self._y


class Numberer:
	def __init__(self):
		self.v2n = dict()
		self.n2v = list()
		self.start_idx = 1

	def number(self, value, add_if_absent=True):
		n = self.v2n.get(value)

		if n is None:
			if add_if_absent:
				n = len(self.n2v) + self.start_idx
				self.v2n[value] = n
				self.n2v.append(value)
			else:
				return 0

		return n

	def value(self, number):
		# self.n2v[number]
		return tf.gather(self.n2v, number)

	def max_number(self):
		return len(self.n2v) + 1


def preprocess(text):
	# @ striping
	text = re.sub(r"@[^\s]+", "", text)
	text = re.sub(r"^[^\d]$", "", text)
	# parse hashtags
	if not re.match(r"#[^\s]+", text) == None:
		value = re.match(r"#[^\s]+", text).group()
		text = re.sub(r"#[^\s]+", tw.segment(value), text)
	# USER removal
	text = text.replace("@USER", "")
	# URL removal 
	text = text.replace("URL", "")
	return text


def read_lexicon(filename):
	with open(filename, "r") as f:
		lex = {}
		for line in f:
			fields = line.split("\t")
			if len(fields) > 1 :
				# lex[preprocess(fields[1])] = {"Task1":{fields[2].strip():1.0}, "Task2":{fields[3].strip():1.0}, "Task3":{fields[4].strip():1.0}}
				lex[preprocess(fields[1])] = {"Task1":{fields[2].strip():1.0}}
		return lex


def recode_lexicon(lexicon, words, labels, train=False):
	int_lex_task1 = []
	int_lex_task2 = []
	int_lex_task3 = []
	with open ("off.txt", "r+") as badwords:
		with open ("offNew.txt", "w+") as badwordswrite:
			for line in badwords.readlines():
				if len(line.strip()) > 0:
					badwordswrite.write(str(words.number(line.strip(), train)) + " \n")
	for (sentence, tags) in lexicon.items():
		int_sentence = []
		for word in sentence.split():
			int_sentence.append(words.number(word, train))
		int_tags_task1 = {}
		int_tags_task2 = {}
		int_tags_task3 = {}
		for (tag, p) in tags["Task1"].items():
			int_tags_task1[labels.number(tag, train)] = p
		# for (tag, p) in tags["Task2"].items():
		# 	int_tags_task2[labels.number(tag, train)] = p
		# for (tag, p) in tags["Task3"].items():
		# 	int_tags_task3[labels.number(tag, train)] = p
		int_lex_task1.append((int_sentence, int_tags_task1))
		# int_lex_task2.append((int_sentence, int_tags_task2))
		# int_lex_task3.append((int_sentence, int_tags_task3))
	# return [int_lex_task1, int_lex_task2, int_lex_task3]
	return [int_lex_task1]


def generate_instances(
		data,
		max_label,
		max_time_steps,
		size_batch=128):
	n_batches = len(data) // size_batch
	# We are discarding the last batch for now, for simplicity.
	labels = np.zeros(
		shape=(
			n_batches,
			size_batch,
			max_label.max_number()),
		dtype=np.float32)
	lengths = np.zeros(
		shape=(
			n_batches,
			size_batch),
		dtype=np.int32)
	sentences = np.zeros(
		shape=(
			n_batches,
			size_batch,
			max_time_steps),
		dtype=np.int32)

	for batch in range(n_batches):
		for idx in range(size_batch):
			(sentence, l) = data[(batch * size_batch) + idx]
			# Add label distribution
			for label, prob in l.items():
				labels[batch, idx, label] = prob
			# Sequence
			timesteps = min(max_time_steps, len(sentence))
			# Sequence length (time steps)
			lengths[batch, idx] = timesteps
			# Word characters
			sentences[batch, idx, :timesteps] = sentence[:timesteps]
	return (sentences, lengths, labels)


def train_model(config, train_batches, validation_batches, numberer):
	train_batches, train_lens, train_labels = train_batches
	validation_batches, validation_lens, validation_labels = validation_batches
	n_chars = max(np.amax(validation_batches), np.amax(train_batches)) + 1
	tf.reset_default_graph()
	with tf.Session() as sess:
		with tf.variable_scope("model", reuse=False):
			train_model = Model(
				config,
				train_batches,
				train_lens,
				train_labels,
				n_chars,
				numberer,
				phase=Phase.Train)

		with tf.variable_scope("model", reuse=True):
			validation_model = Model(
				config,
				validation_batches,
				validation_lens,
				validation_labels,
				n_chars,
				numberer,
				phase=Phase.Validation)
		sess.run(tf.global_variables_initializer())
		print()
		print("       | train   |                   validation                    |")
		print(" epoch | loss    | loss    | acc.    | prec.   | recall  | F1      |")
		print("-------+---------+---------+---------+---------+---------+---------+")
		for epoch in range(config.num_epochs):
			train_loss = 0.0
			validation_loss = 0.0
			accuracy = 0.0
			precision = 0.0
			recall = 0.0
			f1 = 0.0

			# Train on all batches.
			for batch in range(train_batches.shape[0]):
				loss, _ = sess.run([train_model.loss, train_model.train_op], {
					train_model.x: train_batches[batch], train_model.lens: train_lens[batch], train_model.y: train_labels[batch]})
				train_loss += loss

			# validation on all batches.
			with open ("offNew.txt", "r+") as off:
				for batch in range(validation_batches.shape[0]):
					for element in off.readlines():
						for item in range(len(validation_batches[batch])):
							if int(element) in validation_batches[batch][item]:
								validation_labels[batch][item][1] = 1
								
					
						
					loss, acc, hpl = sess.run([validation_model.loss, validation_model.accuracy, validation_model.hp_labels], {
						validation_model.x: validation_batches[batch], validation_model.lens: validation_lens[batch], validation_model.y: validation_labels[batch]})
					
					
					validation_loss += loss
					accuracy += acc
					precision += metrics.precision_score(np.argmax(np.array(validation_labels[batch]).astype(np.int32), axis = 1), hpl, average = "macro")
					recall += metrics.recall_score(np.argmax(np.array(validation_labels[batch]).astype(np.int32), axis = 1), hpl, average = "macro")
					f1 += metrics.f1_score(np.argmax(np.array(validation_labels[batch]).astype(np.int32), axis = 1), hpl, average = "macro")

			train_loss /= train_batches.shape[0]
			validation_loss /= validation_batches.shape[0]
			accuracy /= validation_batches.shape[0]
			precision /= validation_batches.shape[0]
			recall /= validation_batches.shape[0]
			f1 /= validation_batches.shape[0]

			print(" % 3d   | % 4.2f | % 4.2f | % 2.2f%% | % 2.2f%% | % 2.2f%% | % 2.2f%% |" % (epoch, train_loss, validation_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))


if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.stderr.write("Please Use the format: %s TRAINING_SET TEST_SET\n" % sys.argv[0])
		sys.exit(1)

	config = Configuration()

	# Read training, validation, and embedding data.
	train_lexicon = read_lexicon(sys.argv[1])
	validation_lexicon = read_lexicon(sys.argv[2])
	
	# Convert word characters and part-of-speech labels to numeral representation.
	words = Numberer()
	labels = Numberer()

	train_lexicon = recode_lexicon(train_lexicon, words, labels, train=True)
	validation_lexicon = recode_lexicon(validation_lexicon, words, labels)

	# Generate batches
	train_batches = generate_instances(
		train_lexicon[0],
		labels,
		config.max_time_steps,
		size_batch=config.size_batch)
	
	validation_batches = generate_instances(
		validation_lexicon[0],
		labels,
		config.max_time_steps,
		size_batch=config.size_batch)


	# Train the model
	train_model(config, train_batches, validation_batches, words)

	# # Generate batches
	# train_batches = generate_instances(
	# 	train_lexicon[1],
	# 	labels,
	# 	config.max_time_steps,
	# 	size_batch=config.size_batch)
	
	# validation_batches = generate_instances(
	# 	validation_lexicon[1],
	# 	labels,
	# 	config.max_time_steps,
	# 	size_batch=config.size_batch)

	# # Train the model
	# train_model(config, train_batches, validation_batches, words)

	# # Generate batches
	# train_batches = generate_instances(
	# 	train_lexicon[2],
	# 	labels,
	# 	config.max_time_steps,
	# 	size_batch=config.size_batch)
	
	# validation_batches = generate_instances(
	# 	validation_lexicon[2],
	# 	labels,
	# 	config.max_time_steps,
	# 	size_batch=config.size_batch)

	# Train the model
	# train_model(config, train_batches, validation_batches, words)
