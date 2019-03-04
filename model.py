import tensorflow as tf
from parameters import FLAGS
import numpy as np

class network(object):



	############################################################################################################################
	def __init__(self, embeddings_char, embeddings_word):

		with tf.device('/device:GPU:0'):

			# create word embeddings
			self.tf_embeddings = tf.Variable(tf.constant(0.0, shape=[embeddings_word.shape[0], embeddings_word.shape[1]]), trainable=False, name="tf_embeddings")
			self.embedding_placeholder = tf.placeholder(tf.float32, [embeddings_word.shape[0], embeddings_word.shape[1]])
			self.embedding_init = self.tf_embeddings.assign(self.embedding_placeholder)  # initialize this once  with sess.run when the session begins


			# create GRU cells
			with tf.variable_scope("tweet"):
				self.cell_fw = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.rnn_cell_size, activation=tf.sigmoid)
				self.cell_bw = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.rnn_cell_size, activation=tf.sigmoid)

			#placeholders
			self.reg_param = tf.placeholder(tf.float32, shape=[])
			self.Y = tf.placeholder(tf.float64, [FLAGS.batch_size, FLAGS.num_classes])

			num_of_total_filters = len(FLAGS.filter_sizes.split(",")) * FLAGS.num_filters
			total_tweets = FLAGS.batch_size * FLAGS.tweet_per_user

			# weigths
			self.weights = {'fc1': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size, FLAGS.num_classes]), name="fc1-weights"),
					'att1-w': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size, 2 * FLAGS.rnn_cell_size]), name="att1-weights"), #rnn word level
					'att1-v': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size]), name="att1-vector"),							  #rnn word level
					'att2-w': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size, 2 * FLAGS.rnn_cell_size]), name="att2-weights"), #rnn user level
					'att2-v': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size]), name="att2-vector"),							  #rnn user level
					'att2-cnn-w': tf.Variable(tf.random_normal([num_of_total_filters, num_of_total_filters]), name="att2-weights"),	  #cnn user level
					'att2-cnn-v': tf.Variable(tf.random_normal([num_of_total_filters]), name="att2-vector"),						  #cnn user level
					'att3-fusion-w': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size, 2 * FLAGS.rnn_cell_size]), name="att3-weights"), #fusion	  
					'att3-fusion-v': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size]), name="att3-vector")}						     #fusion
			# biases
			self.bias = {'fc1': tf.Variable(tf.random_normal([FLAGS.num_classes]), name="fc1-bias-noreg"),
				     'att1-w': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size]), name="att1-bias-noreg"),  #rnn word level
				     'att2-w': tf.Variable(tf.random_normal([2 * FLAGS.rnn_cell_size]), name="att2-bias-noreg"),  #rnn user level
				     'att2-cnn-w': tf.Variable(tf.random_normal([num_of_total_filters]), name="att2-bias-noreg"), #cnn user level
					 'att3-fusion-w': tf.Variable(tf.random_normal([num_of_total_filters]), name="att3-bias-noreg")} #fusion


			# initialize the computation graph for the neural network
			self.rnn_with_attention()
			self.cnn(embeddings_char.shape[0])
			self.architecture()
			self.backward_pass()








    ############################################################################################################################
	def architecture(self):

		with tf.device('/device:GPU:0'):
			#user level attention
			self.att_context_vector_char = tf.tanh(tf.tensordot(self.cnn_output, self.weights["att2-cnn-w"], axes=1) + self.bias["att2-cnn-w"])
			self.attentions_char = tf.nn.softmax(tf.tensordot(self.att_context_vector_char, self.weights["att2-cnn-v"], axes=1))
			self.attention_output_cnn = tf.reduce_sum(self.cnn_output * tf.expand_dims(self.attentions_char, -1), 1)

			#user level attention
			self.att_context_vector_word = tf.tanh(tf.tensordot(self.rnn_output, self.weights["att2-w"], axes=1) + self.bias["att2-w"])
			self.attentions_word = tf.nn.softmax(tf.tensordot(self.att_context_vector_word, self.weights["att2-v"], axes=1))
			self.attention_output_rnn = tf.reduce_sum(self.rnn_output * tf.expand_dims(self.attentions_word, -1), 1)


			#fusion of rnn and cnn
			self.temp = tf.expand_dims(self.attention_output_rnn, 1)
			self.temp2 = tf.expand_dims(self.attention_output_cnn, 1)
			self.concat_output = tf.concat([self.temp, self.temp2], 1)

			self.att_context_vector_fusion = tf.tanh(tf.tensordot(self.concat_output, self.weights["att3-fusion-w"], axes=1) + self.bias["att3-fusion-w"])
			self.attentions_fusion = tf.nn.softmax(tf.tensordot(self.att_context_vector_fusion, self.weights["att3-fusion-v"], axes=1))
			self.attention_output_fusion = tf.reduce_sum(self.concat_output * tf.expand_dims(self.attentions_fusion, -1), 1)


			# FC layer for reducing the dimension to 2(# of classes)
			self.logits = tf.matmul(self.attention_output_fusion, self.weights["fc1"]) + self.bias["fc1"]

			# predictions
			self.prediction = tf.nn.softmax(self.logits)

			# calculate accuracy
			self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

			return self.prediction






    ############################################################################################################################
	def backward_pass(self):

		with tf.device('/device:GPU:0'):
			# calculate loss
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))

			# add L2 regularization
			self.l2 = self.reg_param * sum(
				tf.nn.l2_loss(tf_var)
				for tf_var in tf.trainable_variables()
				if not ("noreg" in tf_var.name or "bias" in tf_var.name)
			)
			self.loss += self.l2

			# optimizer
			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			self.train = self.optimizer.minimize(self.loss)

			return self.accuracy, self.loss, self.train









############################################################################################################################
	def rnn_with_attention(self):

		with tf.device('/device:GPU:0'):

			#rnn placeholders
			self.X = tf.placeholder(tf.int32, [FLAGS.batch_size*FLAGS.tweet_per_user, None])
			self.sequence_length = tf.placeholder(tf.int32, [FLAGS.batch_size*FLAGS.tweet_per_user])

			# embedding layer
			self.rnn_input = tf.nn.embedding_lookup(self.tf_embeddings, self.X)

			# rnn layer
			(self.outputs, self.output_states) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, self.rnn_input, self.sequence_length, dtype=tf.float32,scope="tweet")

			# concatenate the backward and forward cells
			self.concat_outputs = tf.concat(self.outputs, 2)

			# attention layer
			self.att_context_vector = tf.tanh(tf.tensordot(self.concat_outputs, self.weights["att1-w"], axes=1) + self.bias["att1-w"])
			self.attentions = tf.nn.softmax(tf.tensordot(self.att_context_vector, self.weights["att1-v"], axes=1))
			self.attention_output_raw = tf.reduce_sum(self.concat_outputs * tf.expand_dims(self.attentions, -1), 1)

			#reshape the output for the next layers
			self.rnn_output = tf.reshape(self.attention_output_raw, [FLAGS.batch_size, FLAGS.tweet_per_user, 2*FLAGS.rnn_cell_size])

			return self.rnn_output










############################################################################################################################
	def cnn(self, vocab_size):

		with tf.device('/device:GPU:0'):

			# CNN placeholders
			self.input_x = tf.placeholder(tf.int32, [FLAGS.batch_size*FLAGS.tweet_per_user, FLAGS.sequence_length], name="input_x")

			filter_sizes = [int(size) for size in FLAGS.filter_sizes.split(",")]

			# Embedding layer
			with tf.name_scope("embedding"):
				W = tf.Variable(tf.random_uniform([vocab_size, FLAGS.char_embedding_size], -1.0, 1.0), name="W")
				self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
				self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

			# Create a convolution + maxpool layer for each filter size
			pooled_outputs = []
			for i, filter_size in enumerate(filter_sizes):
				with tf.name_scope("conv-maxpool-%s" % filter_size):
					# Convolution Layer
					filter_shape = [filter_size, FLAGS.char_embedding_size, 1, FLAGS.num_filters]
					W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
					b = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_filters]), name="b-noreg")
					conv = tf.nn.conv2d(
					self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
					# Apply nonlinearity
					h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
					# Maxpooling over the outputs
					pooled = tf.nn.max_pool(
						h,
						ksize=[1, FLAGS.sequence_length - filter_size + 1, 1, 1],
						strides=[1, 1, 1, 1],
						padding='VALID',
						name="pool")
					pooled_outputs.append(pooled)

			# Combine all the pooled features
			num_filters_total = FLAGS.num_filters * len(filter_sizes)
			self.h_pool = tf.concat(pooled_outputs, 3)
			self.h_flat_pool = tf.reshape(self.h_pool, [-1, num_filters_total])

			self.cnn_output = tf.reshape(self.h_flat_pool, [FLAGS.batch_size, FLAGS.tweet_per_user, num_filters_total])

			return self.cnn_output


















