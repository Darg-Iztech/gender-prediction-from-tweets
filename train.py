from parameters import FLAGS
import tensorflow as tf
from preprocess import *
import numpy as np
from model import network



###########################################################################################################################
##trains and validates the model
###########################################################################################################################
def train(network, training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, target_values, vocabulary_word, vocabulary_char, embeddings_char, embeddings_word):

	saver = tf.train.Saver(max_to_keep=None)

	with tf.device('/device:GPU:0'):
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

			# init variables
			init = tf.global_variables_initializer()
			sess.run(init)
			sess.run(network.embedding_init, feed_dict={network.embedding_placeholder: embeddings_word})


			#load the model from checkpoint file if it is required
			if FLAGS.use_pretrained_model == True:
				load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
				saver.restore(sess, load_as)
				print("Loading the pretrained model from: " + str(load_as))


			#for each epoch
			for epoch in range(FLAGS.num_epochs):
				epoch_loss = 0.0
				epoch_accuracy = 0.0
				num_batches = 0.0            
				batch_accuracy = 0.0
				batch_loss = 0.0
				training_batch_count = int(len(training_tweets) / (FLAGS.batch_size*FLAGS.tweet_per_user))
				valid_batch_count = int(len(valid_tweets) / (FLAGS.batch_size*FLAGS.tweet_per_user))

				#TRAINING
				for batch in range(training_batch_count):

					#prepare the batch
					training_batch_x_rnn, training_batch_y_rnn, training_batch_seqlen = prepWordBatchData(training_tweets, training_users, target_values, training_seq_lengths, batch)
					training_batch_x_rnn = word2id(training_batch_x_rnn, vocabulary_word)

					training_batch_x_cnn, training_batch_y_cnn = prepCharBatchData(training_tweets, training_users, target_values, batch)					
					training_batch_x_cnn = char2id(training_batch_x_cnn, vocabulary_char)

					#Flatten everything to feed RNN
					training_batch_x_rnn = np.reshape(training_batch_x_rnn, (FLAGS.batch_size*FLAGS.tweet_per_user, np.shape(training_batch_x_rnn)[2]))
					training_batch_seqlen = np.reshape(training_batch_seqlen, (-1)) # to flatten list, pass [-1] as shape

					#Flatten everything to feed CNN
					training_batch_x_cnn = np.reshape(training_batch_x_cnn, (FLAGS.batch_size*FLAGS.tweet_per_user, FLAGS.sequence_length))


					#run the graph
					feed_dict = {network.input_x: training_batch_x_cnn,\
							 	network.X: training_batch_x_rnn, network.sequence_length: training_batch_seqlen,\
							 	network.Y: training_batch_y_rnn, network.reg_param: FLAGS.l2_reg_lambda}

					_, loss, prediction, accuracy = sess.run([network.train, network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)
					#a, b, c, d = sess.run([network.attention_output_rnn, network.temp, network.temp2, network.concat_output], feed_dict=feed_dict)
					#print np.shape(a)
					#print np.shape(b)
					#print np.shape(c)
					#print np.shape(d)
					#sys.exit()
					#calculate the metrics
					batch_loss += loss
					epoch_loss += loss
					batch_accuracy += accuracy
					epoch_accuracy += accuracy
					num_batches += 1
					
					#print the accuracy and progress of the training
					if batch % FLAGS.evaluate_every == 0 and batch != 0:
						batch_accuracy /= num_batches
						print("Epoch " +"{:2d}".format(epoch)+ " , Batch " +"{0:5d}".format(batch)+ "/" +str(training_batch_count)+ " , loss= " +"{0:5.4f}".format(batch_loss)+ 
							" , accuracy= " + "{0:0.5f}".format(batch_accuracy))
						batch_loss = 0.0
						batch_accuracy = 0.0
						num_batches = 0.0
					
				

				#VALIDATION     
				batch_accuracy = 0.0
				batch_loss = 0.0

				for batch in range(valid_batch_count):

					#prepare the batch
					valid_batch_x_rnn, valid_batch_y_rnn, valid_batch_seqlen = prepWordBatchData(valid_tweets, valid_users, target_values, valid_seq_lengths, batch)
					valid_batch_x_rnn = word2id(valid_batch_x_rnn, vocabulary_word)

					valid_batch_x_cnn, valid_batch_y = prepCharBatchData(valid_tweets, valid_users, target_values, batch)
					valid_batch_x_cnn = char2id(valid_batch_x_cnn, vocabulary_char)

					#Flatten everything to feed RNN
					valid_batch_x_rnn = np.reshape(valid_batch_x_rnn, (FLAGS.batch_size*FLAGS.tweet_per_user, np.shape(valid_batch_x_rnn)[2]))
					valid_batch_seqlen = np.reshape(valid_batch_seqlen, (-1)) # to flatten list, pass [-1] as shape

					#Flatten everything to feed CNN
					valid_batch_x_cnn = np.reshape(valid_batch_x_cnn, (FLAGS.batch_size*FLAGS.tweet_per_user, FLAGS.sequence_length))


					#run the graph
					feed_dict = {network.input_x: valid_batch_x_cnn,\
							 network.X: valid_batch_x_rnn, network.sequence_length: valid_batch_seqlen,\
							 network.Y: valid_batch_y_rnn, network.reg_param: FLAGS.l2_reg_lambda}

					loss, prediction, accuracy = sess.run([network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)


					#calculate the metrics
					batch_loss += loss
					batch_accuracy += accuracy

				#print the accuracy and progress of the validation
				batch_accuracy /= valid_batch_count
				epoch_accuracy /= training_batch_count
				print("Epoch " + str(epoch) + " training loss: " + "{0:5.4f}".format(epoch_loss))
				print("Epoch " + str(epoch) + " training accuracy: " + "{0:0.5f}".format(epoch_accuracy))
				print("Epoch " + str(epoch) + " validation loss: " + "{0:5.4f}".format(batch_loss))
				print("Epoch " + str(epoch) + " validation accuracy: " + "{0:0.5f}".format(batch_accuracy))

				if FLAGS.optimize:
					f = open(FLAGS.log_path, "a")

					training_loss_line = "Epoch " + str(epoch) + " training loss: " + str(epoch_loss) + "\n"
					training_accuracy_line = "Epoch " + str(epoch) + " training accuracy: " + str(epoch_accuracy) + "\n"
					validation_loss_line = "Epoch " + str(epoch) + " validation loss: " + str(batch_loss) + "\n"
					validation_accuracy_line = "Epoch " + str(epoch) + " validation accuracy: " + str(batch_accuracy) + "\n"


					f.write(training_loss_line)
					f.write(training_accuracy_line)
					f.write(validation_loss_line)
					f.write(validation_accuracy_line)

					f.close()


				#save the model if it performs above the threshold
				#naming convention for the model : {"language"}-model-{"learning rate"}-{"reg. param."}-{"epoch number"}
				if batch_accuracy >= FLAGS.model_save_threshold:
					model_name = str(FLAGS.lang) + "-model-" + str(FLAGS.rnn_cell_size) + "-" + str(FLAGS.num_filters) + "-" \
										+str(FLAGS.learning_rate) + "-" + str(FLAGS.l2_reg_lambda) + "-" + str(epoch) + ".ckpt"
					save_as = os.path.join(FLAGS.model_path, model_name)
					save_path = saver.save(sess, save_as)
					print("Model saved in path: %s" % save_path)











####################################################################################################################
#main function for standalone runs
####################################################################################################################
if __name__ == "__main__":

	print("---PREPROCESSING STARTED---")

	print("\treading word embeddings...")
	vocabulary_word, embeddings_word = readGloveEmbeddings(FLAGS.word_embed_path, FLAGS.word_embedding_size)

	print("\treading char embeddings...")
	vocabulary_char, embeddings_char = readCharEmbeddings(FLAGS.char_embed_path, FLAGS.char_embedding_size)

	print("\treading tweets...")
	tweets, users, target_values, seq_lengths = readData(FLAGS.training_data_path)

	print("\tconstructing datasets and network...")
	training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, _, _, _ = partite_dataset(tweets, users, seq_lengths)


	#single run on training data
	if FLAGS.optimize == False:
		#print specs
		print("---TRAINING STARTED---")
		model_specs = "with parameters: Learning Rate:" + str(FLAGS.learning_rate) + ", Regularization parameter:" + str(FLAGS.l2_reg_lambda) + ", fully connected size:"
		model_specs+=  str(FLAGS.fc_size) + ", language:" + FLAGS.lang
		print(model_specs)

		#run the network
		tf.reset_default_graph()
		net = network(embeddings_char, embeddings_word)
		train(net, training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, \
				target_values, vocabulary_word, vocabulary_char, embeddings_char, embeddings_word)

	#hyperparameter optimization
	else:
		for i in range(len(FLAGS.rnn_cell_sizes)):
			for learning_rate in FLAGS.l_rate:
				for regularization_param in FLAGS.reg_param:

					#prep the network
					tf.reset_default_graph()
					FLAGS.learning_rate = learning_rate
					FLAGS.l2_reg_lambda = regularization_param
					FLAGS.rnn_cell_size = FLAGS.rnn_cell_sizes[i]
					FLAGS.num_filters = FLAGS.cnn_filter_counts[i]
					net = network(embeddings_char, embeddings_word)

					#print specs
					print("---TRAINING STARTED---")
					model_specs = "with parameters: Learning Rate:" + str(FLAGS.learning_rate) + ", Regularization parameter:" + str(FLAGS.l2_reg_lambda) + ", rnn cell size:"
					model_specs+=  str(FLAGS.rnn_cell_size) + ", filter size:" + str(FLAGS.num_filters) + ", language:" + FLAGS.lang
					print(model_specs)

					#take the logs
					f = open(FLAGS.log_path,"a")
					f.write("---TRAINING STARTED---\n")
					model_specs += "\n"
					f.write(model_specs)
					f.close()

					#start training
					train(net, training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, \
						target_values, vocabulary_word, vocabulary_char, embeddings_char, embeddings_word)





