#!/usr/bin/python3

# Authors:	Peter Schoener, 4013996
#			Alon Borenstein, 4041104
#			Daniel Nagel, 3098420
# Honor Code: We pledge that this program represents our own work.

from enum import Enum
import sys
import re

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn import metrics

class DefaultConfig:
	n_epochs = 20
	batch_size = 512
	max_timesteps = 40
	LSTM_ct = 4
	LSTM_sz = 200
	dropout_ratio = 0.95
	embedding_sz = 100
	learning_rate = 0.001

class Phase(Enum):
	Train = 0
	Validation = 1
	Predict = 2

class Model:
	def __init__(self, config, batch, lens_batch, label_batch, n_chars, numberer, phase = Phase.Predict):
		batch_size = batch.shape[1]
		input_size = batch.shape[2]
		label_size = label_batch.shape[2]
		
		# The integer-encoded words. input_size is the (maximum) number of
		# time steps.
		self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

		# This tensor provides the actual number of time steps for each
		# instance.
		self._lens = tf.placeholder(tf.int32, shape=[batch_size])

		# The label distribution.
		if phase != Phase.Predict:
			self._y = tf.placeholder(
				tf.float32, shape=[batch_size, label_size])

		# convert to embeddings
		embeddings = tf.get_variable("embeddings", shape = [n_chars, config.embedding_sz])
		input_layer = tf.nn.embedding_lookup(embeddings, self._x)

		# make a bunch of LSTM cells and link them
		# use rnn.DropoutWrapper instead of tf.nn.dropout because the layers are anonymous
		stacked_LSTM = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(config.LSTM_sz), output_keep_prob = config.dropout_ratio) for _ in range(config.LSTM_ct)])
				
		# run the whole thing
		_, hidden = tf.nn.dynamic_rnn(stacked_LSTM, input_layer, sequence_length = self._lens, dtype = tf.float32)
		w = tf.get_variable("W", shape=[hidden[-1].h.shape[1], label_size]) # if I understood the structure of MultiRNNCell correctly, hidden[-1] should be the final state
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
	# convert to lowercase
	text = text.lower()
	# strip URLs (imperfect)
	text = re.sub(r"((www\.[^\s]+)|(https?://[^\s]+))", "", text)
	# strip @ mentions
	text = re.sub(r"@[^\s]+", "", text)
	# replace hashtags
	text = text.replace("#", "")
	
	return text


def read_lexicon(filename):
	with open(filename, "r") as f:
		lex = {}
		
		for line in f:
			fields = line.split("\t")
			if len(fields) > 3 :
				lex[preprocess(fields[1])] = {fields[3].strip():1.0}
		return lex


def recode_lexicon(lexicon, words, labels, train=False):
	int_lex = []

	for (sentence, tags) in lexicon.items():
		int_sentence = []
		for word in sentence.split():
			int_sentence.append(words.number(word, train))

		int_tags = {}
		for (tag, p) in tags.items():
			int_tags[labels.number(tag, train)] = p

		int_lex.append((int_sentence, int_tags))

	return int_lex


def generate_instances(
		data,
		max_label,
		max_timesteps,
		batch_size=128):
	n_batches = len(data) // batch_size

	# We are discarding the last batch for now, for simplicity.
	labels = np.zeros(
		shape=(
			n_batches,
			batch_size,
			max_label.max_number()),
		dtype=np.float32)
	lengths = np.zeros(
		shape=(
			n_batches,
			batch_size),
		dtype=np.int32)
	sentences = np.zeros(
		shape=(
			n_batches,
			batch_size,
			max_timesteps),
		dtype=np.int32)

	for batch in range(n_batches):
		for idx in range(batch_size):
			(sentence, l) = data[(batch * batch_size) + idx]

			# Add label distribution
			for label, prob in l.items():
				labels[batch, idx, label] = prob

			# Sequence
			timesteps = min(max_timesteps, len(sentence))

			# Sequence length (time steps)
			lengths[batch, idx] = timesteps

			# Word characters
			sentences[batch, idx, :timesteps] = sentence[:timesteps]

	return (sentences, lengths, labels)


def train_model(config, train_batches, validation_batches, numberer):
	train_batches, train_lens, train_labels = train_batches
	validation_batches, validation_lens, validation_labels = validation_batches

	n_chars = max(np.amax(validation_batches), np.amax(train_batches)) + 1

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
		for epoch in range(config.n_epochs):
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
			for batch in range(validation_batches.shape[0]):
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
		sys.stderr.write("Usage: %s TRAIN_SET DEV_SET\n" % sys.argv[0])
		sys.exit(1)

	config = DefaultConfig()

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
		train_lexicon,
		labels,
		config.max_timesteps,
		batch_size=config.batch_size)
	validation_batches = generate_instances(
		validation_lexicon,
		labels,
		config.max_timesteps,
		batch_size=config.batch_size)

	# Train the model
	train_model(config, train_batches, validation_batches, words)

# best result:
# 72.90% accuracy
# in this epoch, the model had
# 60.63% precision
# 48.92% recall
# 51.01% F1-score
