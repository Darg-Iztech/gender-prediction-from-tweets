from parameters import FLAGS
import tensorflow as tf
from preprocess import *
from model import network





#####################################################################################################################
##loads a model and tests it
#####################################################################################################################
def test(network, test_tweets, test_users, test_seq_lengths, target_values, vocabulary_word, vocabulary_char, embeddings_char, embeddings_word):
	
	saver = tf.train.Saver(max_to_keep=None)

	with tf.device('/device:GPU:0'):
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

			# init variables
			init = tf.global_variables_initializer()
			sess.run(init)
			sess.run(network.embedding_init, feed_dict={network.embedding_placeholder: embeddings_word})
			batch_loss = 0.0
			batch_accuracy = 0.0


			#load the model from checkpoint file
			load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
			print("Loading the pretrained model from: " + str(load_as))
			saver.restore(sess, load_as)


			#start evaluating each batch of test data
			batch_count = int(len(test_tweets) / (FLAGS.batch_size*FLAGS.tweet_per_user))

			for batch in range(batch_count):

				#prepare the batch
				test_batch_x_rnn, test_batch_y_rnn, test_batch_seqlen = prepWordBatchData(test_tweets, test_users, target_values, test_seq_lengths, batch)
				test_batch_x_rnn = word2id(test_batch_x_rnn, vocabulary_word)

				test_batch_x_cnn, test_batch_y_cnn = prepCharBatchData(test_tweets, test_users, target_values, batch)
				test_batch_x_cnn = char2id(test_batch_x_cnn, vocabulary_char)


				#Flatten everything to feed RNN
				test_batch_x_rnn = np.reshape(test_batch_x_rnn, (FLAGS.batch_size*FLAGS.tweet_per_user, np.shape(test_batch_x_rnn)[2]))
				test_batch_seqlen = np.reshape(test_batch_seqlen, (-1)) # to flatten list, pass [-1] as shape

				#Flatten everything to feed CNN
				test_batch_x_cnn = np.reshape(test_batch_x_cnn, (FLAGS.batch_size*FLAGS.tweet_per_user, FLAGS.sequence_length))



				#run the graph
				feed_dict = {network.input_x: test_batch_x_cnn,\
							 network.X: test_batch_x_rnn, network.sequence_length: test_batch_seqlen,\
							 network.Y: test_batch_y_rnn, network.reg_param: FLAGS.l2_reg_lambda}

				loss, prediction, accuracy = sess.run([network.loss, network.prediction, network.accuracy], feed_dict=feed_dict)


				#calculate the metrics
				batch_loss += loss
				batch_accuracy += accuracy

			#print the accuracy and progress of the validation
			batch_accuracy /= batch_count
			print("Test loss: " + "{0:5.4f}".format(batch_loss))
			print("Test accuracy: " + "{0:0.5f}".format(batch_accuracy))

			if FLAGS.optimize:
				f = open(FLAGS.log_path, "a")
				f.write("\n---TESTING STARTED---\n")			
				f.write("with model:" + load_as + "\n")
				f.write("Test loss: " + "{0:5.4f}".format(batch_loss) + "\n")
				f.write("Test accuracy: " + "{0:0.5f}".format(batch_accuracy) + "\n")
				f.close()














#############################################################################################################
#main function for standalone runs
#############################################################################################################
if __name__ == "__main__":

	print("---PREPROCESSING STARTED---")

	print("\treading word embeddings...")
	vocabulary_word, embeddings_word = readGloveEmbeddings(FLAGS.word_embed_path, FLAGS.word_embedding_size)

	print("\treading char embeddings...")
	vocabulary_char, embeddings_char = readCharEmbeddings(FLAGS.char_embed_path, FLAGS.char_embedding_size)

	print("\treading tweets for test...")
	tweets, users, target_values, seq_lengths = readData(FLAGS.test_data_path)
	print("\ttest set size: " + str(len(tweets)))

	print("---TESTING STARTED---")
	#finds every model in FLAGS.model_path and runs every single one
	if FLAGS.optimize == True:
		models = os.listdir(FLAGS.model_path)
		for model in models:
			if model.endswith(".ckpt.index"):
				FLAGS.model_name = model[:-6]
				tf.reset_default_graph()

				if "90" in FLAGS.model_name:
					FLAGS.num_filters = 60
					FLAGS.rnn_cell_size = 90
				elif "120" in FLAGS.model_name:
					FLAGS.num_filters = 80
					FLAGS.rnn_cell_size = 120
				elif "150" in FLAGS.model_name:
					FLAGS.num_filters = 100
					FLAGS.rnn_cell_size = 150

				net = network(embeddings_char, embeddings_word)
				test(net, tweets, users, seq_lengths, target_values, vocabulary_word, vocabulary_char, embeddings_char, embeddings_word)
	#just runs  single model specified in FLAGS.model_path and FLAGS.model_name
	else:
		tf.reset_default_graph()
		net = network(embeddings_char, embeddings_word)
		test(net, tweets, users, seq_lengths, target_values, vocabulary_word, vocabulary_char, embeddings_char, embeddings_word)




