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
	num_epochs = 500
	size_batch = 43
	max_time_steps = 40
	LSTM_CT = 4
	LSTM_SZ = 200
	ratio_dropout = 0.95
	embedding_size = 100
	rate_learning = 0.005

class PredictionPhase(Enum):
	Training = 0
	Validating = 1
	Prediction = 2

class LSTMModel:
	def __init__(self, configuration, batch_current, lens_batch, label_batch, n_chars, number, phase = PredictionPhase.Prediction):
		size_batch = batch_current.shape[1]
		size_input = batch_current.shape[2]
		size_label = label_batch.shape[2]
		

		# This tensor provides the actual num of time steps for each
		# instance.
		self.self_lens = tf.placeholder(tf.int32, shape=[size_batch])


		# The integer-encoded words. size_input is the (maximum) num of
		# time steps.
		self.self_x = tf.placeholder(tf.int32, shape=[size_batch, size_input])

		
		# The label distribution.
		if phase != PredictionPhase.Prediction:
			self.self_y = tf.placeholder(tf.float32, shape=[size_batch, size_label])

		# convert to embeddings
		embedding = tf.get_variable("embeddings", shape = [n_chars, configuration.embedding_size])
		input_layers = tf.nn.embedding_lookup(embedding, self.self_x)

		# make a bunch of LSTM cells and link them
		# use rnn.DropoutWrapper instead of tf.nn.dropout because the layers are anonymous
		LSTM = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(configuration.LSTM_SZ), output_keep_prob = configuration.ratio_dropout) for _ in range(configuration.LSTM_CT)])
				
		# run the Prediction with variables
		_, hidden_var = tf.nn.dynamic_rnn(LSTM, input_layers, sequence_length = self.self_lens, dtype = tf.float32)
		w = tf.get_variable("W", shape=[hidden_var[-1].h.shape[1], size_label]) 
		
		# Acording to the structure of MultiRNNCell, The hidden_var[-1] is the final state
		b = tf.get_variable("b", shape=[1])
		logits = tf.matmul(hidden_var[-1].h, w) + b

		if phase == PredictionPhase.Training or PredictionPhase.Validating:
			loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.self_y, logits=logits)
			self.self_loss = loss = tf.reduce_sum(loss)

		if phase == PredictionPhase.Training:
			self.self_train_op = tf.train.AdamOptimizer(configuration.rate_learning).minimize(loss)
			self.self_probs = probs = tf.nn.softmax(logits)

		if phase == PredictionPhase.Validating:
			# Highest probability labels of the gold whole_batch_data.
			gs_labels = tf.argmax(self.self_y, axis=1)

			# Predicted labels
			self.self_hp_labels = tf.argmax(logits, axis=1)

			correct = tf.equal(self.self_hp_labels, gs_labels)
			correct = tf.cast(correct, tf.float32)
			self.self_accuracy = tf.reduce_mean(correct)
			
			#self.self_hp_labels = hp_labels

	@property
	def accuracy(self):
		return self.self_accuracy

	@property
	def hp_labels(self) :
		return self.self_hp_labels
	
	@property
	def lens(self):
		return self.self_lens

	@property
	def loss(self):
		return self.self_loss

	@property
	def probs(self):
		return self.self_probs

	@property
	def train_op(self):
		return self.self_train_op

	@property
	def x(self):
		return self.self_x

	@property
	def y(self):
		return self.self_y


class Numbers:
	def __init__(self):
		self.number_to_vector = list()
		self.vector_to_number = dict()
		self.start_index = 1

	def num(self, values, plus_if_not=True):
		n = self.vector_to_number.get(values)

		if n is None:
			if plus_if_not:
				n = len(self.number_to_vector) + self.start_index
				self.vector_to_number[values] = n
				self.number_to_vector.append(values)
			else:
				print (values)
				return 0

		return n

	def values(self, num):
		# self.number_to_vector[num]
		return tf.gather(self.number_to_vector, num)

	def max_num(self):
		return len(self.number_to_vector) + 1


def pre_processing(text):
	# @ striping
	text = re.sub(r"@[^\s]+", "", text)
	# text = re.sub(r"^[^\d]$", "", text)
	# parse hashtags
	if not re.match(r"#[^\s]+", text) == None:
		values = re.match(r"#[^\s]+", text).group()
		text = re.sub(r"#[^\s]+", tw.segment(values), text)
	# USER removal
	text = text.replace("@USER", "")
	# URL removal 
	text = text.replace("URL", "")
	text = text.replace("#", "")
	return text


def lexicon_read(filename, type):
	with open(filename, "r") as f:
		lex = {}
		for line in f:
			fields = line.split("\t")
			if len(fields) > 1 :
				# lex[pre_processing(fields[1])] = {"Task1":{fields[2].strip():1.0}, "Task2":{fields[3].strip():1.0}, "Task3":{fields[4].strip():1.0}}
				if type == "TRAIN":
					lex[fields[0] + " " + pre_processing(fields[1])] = {"Task1":{fields[2].strip():1.0}}
				else:
					lex[fields[0] + " " + pre_processing(fields[1])] = {"Task1":{"OFF":1.0}}
		return lex


def lexicon_recode(lex, words, labels, train=False):
	with open ("off.txt", "r+") as badwords:
		with open ("offNew.txt", "w+") as badwordswrite:
			for line in badwords.readlines():
				if len(line.strip()) > 0:
					badwordswrite.write(str(words.num(line.strip(), train)) + " \n")
	int_lex_task1 = []
	int_lex_task2 = []
	int_lex_task3 = []
	with open ("numbers.txt", "w+") as writing:
		for (line, tags) in lex.items():
			int_sentence = []
			for word in line.split():
				writing.write(word + "\t" + str(words.num(str(word), True)) + "\n")
				int_sentence.append(words.num(word, train))

			int_tags_task1 = {}
			int_tags_task2 = {}
			int_tags_task3 = {}
			for (tag, p) in tags["Task1"].items():
				int_tags_task1[labels.num(tag, train)] = p
			# for (tag, p) in tags["Task2"].items():
			# 	int_tags_task2[labels.num(tag, train)] = p
			# for (tag, p) in tags["Task3"].items():
			# 	int_tags_task3[labels.num(tag, train)] = p
			
			int_lex_task1.append((int_sentence, int_tags_task1))
			
		# int_lex_task2.append((int_sentence, int_tags_task2))
		# int_lex_task3.append((int_sentence, int_tags_task3))
	# return [int_lex_task1, int_lex_task2, int_lex_task3]
	return [int_lex_task1]


def generate_instances( whole_batch_data, max_label, max_time_steps, size_batch=128):
	batch_num = len(whole_batch_data) // size_batch
	# We are discarding the last batch_current for now, for simplicity.
	labels = np.zeros(
		shape=( batch_num, size_batch, max_label.max_num()), dtype=np.float32)
	length = np.zeros(
		shape=( batch_num, size_batch), dtype=np.int32)
	sentences = np.zeros(
		shape=( batch_num, size_batch, max_time_steps), dtype=np.int32)

	for batch_current in range(batch_num):
		for index in range(size_batch):
			(line, l) = whole_batch_data[(batch_current * size_batch) + index]
			for label, prob in l.items(): # Add label distribution
				labels[batch_current, index, label] = prob
			# Sequence
			time_step = min(max_time_steps, len(line))
			# Sequence length (time steps)
			length[batch_current, index] = time_step
			# Word characters
			sentences[batch_current, index, :time_step] = line[:time_step]
	return (sentences, length, labels)


def model_training(configuration, train_batches, validation_batches, number):
	train_batches, train_lens, train_labels = train_batches
	validation_batches, validation_lens, validation_labels = validation_batches
	n_chars = max(np.amax(validation_batches), np.amax(train_batches)) + 1
	tf.reset_default_graph()

	with tf.Session() as sess:
		with tf.variable_scope("model", reuse=False):
			model_training = LSTMModel( configuration, train_batches, train_lens, train_labels, n_chars, number,
				phase=PredictionPhase.Training)

		with tf.variable_scope("model", reuse=True):
			validation_model = LSTMModel( configuration, validation_batches, validation_lens, validation_labels, n_chars, number,
				phase=PredictionPhase.Validating)

		sess.run(tf.global_variables_initializer())

		print("       | Training   |                   Validation                    |")
		print(" Epoch | Train Loss    | Validation Loss    | accuracy    | Precision   | Recall  | F1      |")
		print("-------+---------+---------+---------+---------+---------+---------+")
		for epoch in range(configuration.num_epochs):
			training_loss = 0.0
			validation_loss = 0.0
			accuracy = 0.0
			precision = 0.0
			recall = 0.0
			f1 = 0.0


			all_text = []
			all_labels = []
			for batch_current in range(validation_batches.shape[0]): # validation on all batches.
				loss, acc, hpl = sess.run([validation_model.loss, validation_model.accuracy, validation_model.hp_labels], {
					validation_model.x: validation_batches[batch_current], validation_model.lens: validation_lens[batch_current], validation_model.y: validation_labels[batch_current]})
				validation_loss += loss
				accuracy += acc
				precision += metrics.precision_score(np.argmax(np.array(validation_labels[batch_current]).astype(np.int32), axis = 1), hpl, average = "macro")
				recall += metrics.recall_score(np.argmax(np.array(validation_labels[batch_current]).astype(np.int32), axis = 1), hpl, average = "macro")
				f1 += metrics.f1_score(np.argmax(np.array(validation_labels[batch_current]).astype(np.int32), axis = 1), hpl, average = "macro")
				for item in range(len(hpl.tolist())):
					all_text.append(validation_batches[batch_current][item].tolist())
					all_labels.append(str(hpl.tolist()[item]))

			
			for batch_current in range(train_batches.shape[0]): # Training on all batches.
				loss, _ = sess.run([model_training.loss, model_training.train_op], {
					model_training.x: train_batches[batch_current], model_training.lens: train_lens[batch_current], model_training.y: train_labels[batch_current]})
				training_loss += loss

			training_loss /= train_batches.shape[0]
			validation_loss /= validation_batches.shape[0]
			accuracy /= validation_batches.shape[0]
			precision /= validation_batches.shape[0]
			recall /= validation_batches.shape[0]
			f1 /= validation_batches.shape[0]


			with open ("offNew.txt", "r+") as off:
				for batch in range(validation_batches.shape[0]):
					for element in off.readlines():
						for item in range(len(validation_batches[batch])):
							if int(element) in validation_batches[batch][item]:
								validation_labels[batch][item][1] = 1
							
			print(" % 3d   | % 4.2f | % 4.2f | % 2.2f%% | % 2.2f%% | % 2.2f%% | % 2.2f%% |" % (epoch, training_loss, validation_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))
			with open ("off.txt", "r+") as off:
				with open (str(f1 * 100) + ".txt", "w+") as w:
						with open("numbers.txt", "r+") as f:
							data = set(f.readlines())
							for item in range(len(all_text)):
								tweets = []
								for element in all_text[item]:
									for bit in data:
										if str(element) == str(bit.split("\t")[1].replace("\n", "")):
											tweets.append(bit.split("\t")[0])
											break

								print (off.readlines())
								# for bad_word in off.readlines():
								# 	print (bad_word)
									# print (tweets)
									# if bad_word in tweets:
									# 	print (bad_word)
									# 	print ("bad word")
									# 	print (all_labels[item])
						
								# if str(all_labels[item]) == "0":
								# 	w.write(" ".join(tweets) + " LOSS\n")
								# elif str(all_labels[item]) == "1":
								# 	w.write (" ".join(tweets) + " OFF\n")
								# elif str(all_labels[item]) == "2":
								# 	w.write (" ".join(tweets) + " NOT\n")


if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.stderr.write("Please Use the Both files that are: %s Training Test\n" % sys.argv[0])
		sys.exit(1)

	configuration = Configuration()

	# Convert word characters and part-of-speech labels to numeral representation.
	words = Numbers()
	labels = Numbers()

	# Read training, validation, and embedding whole_batch_data.
	train_lexicon = lexicon_read(sys.argv[1], "TRAIN")
	validation_lexicon = lexicon_read(sys.argv[2], "TEST")
	

	train_lexicon = lexicon_recode(train_lexicon, words, labels, train=True)
	validation_lexicon = lexicon_recode(validation_lexicon, words, labels)

	# Generate batches
	train_batches = generate_instances(
		train_lexicon[0],
		labels,
		configuration.max_time_steps,
		size_batch=configuration.size_batch)
	
	validation_batches = generate_instances(
		validation_lexicon[0],
		labels,
		configuration.max_time_steps,
		size_batch=configuration.size_batch)

	# Training the model
	model_training(configuration, train_batches, validation_batches, words)

	# # Generate batches
	# train_batches = generate_instances(
	# 	train_lexicon[1],
	# 	labels,
	# 	configuration.max_time_steps,
	# 	size_batch=configuration.size_batch)
	
	# validation_batches = generate_instances(
	# 	validation_lexicon[1],
	# 	labels,
	# 	configuration.max_time_steps,
	# 	size_batch=configuration.size_batch)

	# # Training the model
	# model_training(configuration, train_batches, validation_batches, words)

	# # Generate batches
	# train_batches = generate_instances(
	# 	train_lexicon[2],
	# 	labels,
	# 	configuration.max_time_steps,
	# 	size_batch=configuration.size_batch)
	
	# validation_batches = generate_instances(
	# 	validation_lexicon[2],
	# 	labels,
	# 	configuration.max_time_steps,
	# 	size_batch=configuration.size_batch)

	# Training the model
	# model_training(configuration, train_batches, validation_batches, words)
