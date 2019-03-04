import gensim, itertools
import numpy as np
from parameters import FLAGS
import xml.etree.ElementTree as xmlParser
from nltk.tokenize import TweetTokenizer
import os
import sys
import random





#########################################################################################################################
# Read GloVe embeddings
#
# input: String (path)        - Path of embeddings to read
#        int (embedding_size) - Size of the embeddings
#
# output: dict (vocab)             - Dictionary of the vocabulary in GloVe
#         numpy array (embeddings) - Embeddings of the words in GloVe
def readGloveEmbeddings(path, embedding_size):
    DOC_LIMIT = None
    in_file = gensim.models.word2vec.LineSentence(path)

    lines = lambda: itertools.islice(in_file, DOC_LIMIT)
    model_tuple = lambda: ((line[0], [float(value) for value in line[1:]]) for line in lines())

    # extract the keys and values so we can iterate over them
    model_dict = dict(model_tuple())
    temp_vocab = list(model_dict.keys())
    temp_vectors = list(model_dict.values())

    vocab = list()
    vectors = list()
    count = 0

    for line in temp_vectors:
        if len(line) == embedding_size:
            vocab.append(temp_vocab[count])
            vectors.append(temp_vectors[count])
        count += 1
    del temp_vectors, temp_vocab, model_dict

    # add special tokens
    vocab.append("UNK")
    vectors.append(np.random.randn(embedding_size))
    vocab.append("PAD")
    vectors.append(np.zeros(embedding_size))

    embeddings = np.array(vectors)

    vocabulary = {}

    for i in range(len(vocab)):
        vocabulary[vocab[i]] = i

    return vocabulary, embeddings







#########################################################################################################################
# Read GloVe embeddings
#
# input: String (path)        - Path of embeddings to read
#        int (embedding_size) - Size of the embeddings
#
# output: dict (vocab)             - Dictionary of the vocabulary in char embeddings
#         numpy array (embeddings) - Embeddings of the words in char embeddings
def readCharEmbeddings(path, embedding_size):
    in_file = gensim.models.word2vec.LineSentence(path)

    lines = lambda: itertools.islice(in_file, None)

    model_tuple = lambda: ((line[0], [float(value) for value in line[1:]]) for line in lines())
    model_dict = dict(model_tuple())

    char_list = [int(char_number) for char_number in model_dict.keys() if len(model_dict[char_number]) == embedding_size]
    vectors = [embed for embed in model_dict.values() if len(embed) == embedding_size]

    char_list.append(ord('U'))
    vectors.append(np.random.randn(embedding_size))
    char_list.append(ord('P'))
    vectors.append(np.zeros(embedding_size))

    char_vocabulary = {}

    for i in range(len(char_list)):
        char_vocabulary[char_list[i]] = i

    return char_vocabulary, np.array(vectors)







#########################################################################################################################
# Reads training dataset
# one-hot vectors: female = [0,1]
#		           male   = [1,0]
#
# input:  string = path to the zip-file corresponding to the training data
# output: list ("tweets")        = List of tweets
#         list ("users")         = List of users
#	      dict ("target_values") = Author(key) - ground-truth(value) pairs
#	      list ("seq-lengths")   = Lenght of each tweet in the list "training_set"
def readData(path):
    path = os.path.join(os.path.join(path,FLAGS.lang),"text")
    tokenizer = TweetTokenizer()
    training_set = []
    target_values = {}
    seq_lengths = []

    truth_file_name = os.path.join(path,"truth.txt")
    text = open(truth_file_name, 'r')

    # each line = each author
    for line in text:
        words = line.strip().split(':::')
        if words[1] == "female":
            target_values[words[0]] = [0, 1]
        elif words[1] == "male":
            target_values[words[0]] = [1, 0]

    targets = list(target_values.keys())
    np.random.shuffle(targets)

    for user in targets:
        xml_file_name = os.path.join(path,user)
        if sys.version_info[0] < 3:
            xmlFile = open(xml_file_name + ".xml", "r")
        else:
            xmlFile = open(xml_file_name + ".xml", "r", encoding="utf-8")

        rootTag = xmlParser.parse(xmlFile).getroot()

        # for each tweet
        for documents in rootTag:
            for document in documents.findall("document"):
                words = tokenizer.tokenize(document.text)
                training_set.append([user, words])  # author-tweet pairs
                seq_lengths.append(len(words))  # length of tweets will be fed to rnn as timestep size

    tweets = [row[1] for row in training_set]
    users = [row[0] for row in training_set]

    return tweets, users, target_values, seq_lengths








#########################################################################################################################
# Reads captions of images
# one-hot vectors: female = [0,1]
#		           male   = [1,0]
#
# input:  string = path to the txt-file corresponding to the training caption data
# output: list ("sorted_captions")  = List of tweets
#         list ("sorted_authorname") = List of users
#	      dict ("target_values") = Author(key) - ground-truth(value) pairs
#	      list ("seq-lengths")   = Lenght of each tweet in the list "training_set"
def readCaptions(path):
	truth_path = os.path.join(os.path.join(path,FLAGS.lang),"text")
	caption_path = os.path.join(os.path.join(path,FLAGS.lang),"caption")

	tokenizer = TweetTokenizer()
	captions = []
	authorname = []
	target_values = {}
	seq_lengths = []

	#read truth file
	truth_file_name = os.path.join(truth_path,"truth.txt")
	text = open(truth_file_name, 'r')

	# each line = each author
	for line in text:
		words = line.strip().split(':::')
		if words[1] == "female":
			target_values[words[0]] = [0, 1]
		elif words[1] == "male":
			target_values[words[0]] = [1, 0]

	targets = list(target_values.keys())
	np.random.shuffle(targets)
	

	#read caption file
	caption_file_name = os.path.join(caption_path, str(FLAGS.lang) + "_captions.txt")

	with open(caption_file_name) as f:
		for line in f:
			pair = line.strip().split(":::")
			captions.append(pair[1])
			authorname.append(pair[0])
	
	captions = np.array(captions)
	authorname = np.array(authorname)

	
	#find and sort the users' captions
	sorted_captions = []
	sorted_authorname = []

	for user in targets:

		indices = np.where(authorname == user)[0]
		caption_count = len(indices)
		temp_authorname = 0

	
		for index in indices:
			tokenized_Caption = tokenizer.tokenize(captions[index])
			sorted_captions.append(tokenized_Caption)
			sorted_authorname.append(authorname[index])
			seq_lengths.append(len(tokenized_Caption))
			temp_authorname = authorname[index]

		if caption_count < 10:
			while caption_count != 10:
				sorted_captions.append("PAD")
				sorted_authorname.append(temp_authorname)
				seq_lengths.append(1)
				caption_count += 1


	return sorted_captions, sorted_authorname, target_values, seq_lengths








#########################################################################################################################
# Reads vectors of trained models
#
# input:  string(path) = path to the folder that contains the vector files in below format
# format -->   userID:::vector:::truth_value (values in the vector are comma seperated)
#
# output: list(rnn_data) - Vectors of rnn with corrisponding indexes of users
#         list(cnn_data) - Vectors of cnn with corrisponding indexes of users
#         list(image_data) - Vectors of image model with corrisponding indexes of users
#         list(users) - List of usernames
#         list(target_values) - List of target values with corrisponding indexes of users
def readVectors(path):
	rnn_data = []
	users = []
	target_values = {}

	for vector_file in os.listdir(path):
		if "rnn" in vector_file:
			file_handler = open(os.path.join(path,vector_file),"r")
			
			for line in file_handler:
				seperated = line.strip().split(":::")

				users.append(seperated[0]) #add user to user list

				vector_list = [] 
				for value in seperated[1].split(","): #add vector to rnn_data
					vector_list.append(float(value))
				rnn_data.append(vector_list)

				target_values[seperated[0]] = [int(seperated[2].split(",")[0]), int(seperated[2].split(",")[1])] #target values extraction
				

			file_handler.close()

	cnn_data = [i for i in range(len(rnn_data))]
	image_data = [i for i in range(len(rnn_data))]

	for vector_file in os.listdir(path):
		if "rnn" not in vector_file:
			file_handler = open(os.path.join(path,vector_file),"r")

			for line in file_handler:
				seperated = line.strip().split(":::")

				vector_list = []
				for value in seperated[1].split(","):
					vector_list.append(float(value))	
				
				index = users.index(seperated[0])
				if "cnn" in vector_file:
					cnn_data[index] = vector_list
				else:
					image_data[index] = vector_list

			file_handler.close()


	return rnn_data, cnn_data, image_data, users, target_values



#########################################################################################################################
# Shuffles dataset and partites it into 2 part training, validation
#
# input: list (rnn_vectors)  - List of rnn vectors corrisponding to the username indexes
#	     list (cnn_vectors)   - List of cnn vectors corrisponding to the username indexes
#	     list (image_vectors) - List of image model vectors corrisponding to the username indexes
#            list (users) - List of usernames
#
# output: too long, the return line can easily be understood i suppose
def partite_dataset_vectors(rnn_vectors, cnn_vectors, image_vectors, users):

	c = list(zip(rnn_vectors, cnn_vectors, image_vectors, users))
	random.shuffle(c)
	rnn_vectors, cnn_vectors, image_vectors, users = zip(*c)

	rnn_vectors = list(rnn_vectors)
	cnn_vectors = list(cnn_vectors)
	image_vectors = list(image_vectors)
	users = list(users)

	training_set_size = int(len(users) * FLAGS.training_set_size)

	training_rnn_vectors = rnn_vectors[:training_set_size]
	valid_rnn_vectors = rnn_vectors[training_set_size:]

	training_cnn_vectors = cnn_vectors[:training_set_size]
	valid_cnn_vectors = cnn_vectors[training_set_size:]

	training_image_vectors = image_vectors[:training_set_size]
	valid_image_vectors = image_vectors[training_set_size:]

	training_users = users[:training_set_size]
	valid_users = users[training_set_size:]
				

	return training_rnn_vectors, training_cnn_vectors, training_image_vectors, training_users, valid_rnn_vectors, valid_cnn_vectors, valid_image_vectors, valid_users



#########################################################################################################################
# Prepares batch for vectors and shuffles the batch
#
# input:     list (rnn_vectors)  - Vectors of RNN model
#	     list (cnn_vectors)   - Vectors of CNN model
#	     list (image_vectors) - Vectors from image model with corrisponding order in username list
#            list (users) - The usernames
#	     list (target_values) - Ground-truth gender vector of each owner
#	     int  (batch_index) - Current # of iteration we are on
#
# output:     list (batch_rnn_vectors)  - rnn vectors for this batch
# 	      list (batch_cnn_vectors)  - cnn vectors for this batch
#	      list (batch_image_vectors)  - image vectors for this batch
#             list (batch_users)  - The usernames for this specific batch
#             list (batch_targets)  - Corrisponding target values of users for this batch
def prepVectorBatchData(rnn_vectors, cnn_vectors, image_vectors, users, target_values, batch_index):

	start = FLAGS.batch_size * batch_index
	end = FLAGS.batch_size * batch_index + FLAGS.batch_size

	if end > len(users):
		end = len(users)

	batch_rnn_vectors = rnn_vectors[start:end]
	batch_cnn_vectors = cnn_vectors[start:end]
	batch_image_vectors = image_vectors[start:end]
	batch_users = users[start:end]
	batch_targets = user2target(batch_users, target_values)

	
	c = list(zip(batch_rnn_vectors, batch_cnn_vectors, batch_image_vectors, batch_users, batch_targets))
	random.shuffle(c)
	batch_rnn_vectors, batch_cnn_vectors, batch_image_vectors, batch_users, batch_targets = zip(*c)	

	return batch_rnn_vectors, batch_cnn_vectors, batch_image_vectors, batch_users, batch_targets



#########################################################################################################################
# Prepares test data
#
# input: List (tweets)  - List of tweets of a user, each tweet has words as list
#        List (user)    - List of usernames
#        dict (target)  - Dictionary for one-hot gender vectors of users
#
# output: List (test_input)  - List of tweets which are padded up to max_tweet_length
#         List (test_output) - List of one-hot gender vector corresponding to tweets in index order
def prepTestData(tweets, user, target):
    # prepare output
    test_output = user2target(user, target)

    # prepare input by adding padding
    tweet_lengths = [len(tweet) for tweet in tweets]
    max_tweet_length = max(tweet_lengths)

    test_input = []
    for i in range(len(tweets)):
        tweet = tweets[i]
        padded_tweet = []
        for j in range(max_tweet_length):
            if len(tweet) > j:
                padded_tweet.append(tweet[j])
            else:
                padded_tweet.append("PAD")
        test_input.append(padded_tweet)

    return test_input, test_output






#########################################################################################################################
# Returns the one-hot gender vectors of users in correct order (index matching)
#
# input: list (users)   - List of usernames
#        dict (targets) - Dictionary of username(key) and one-hot gender vector(value)
#
# output: list (target_values) - List of one-hot gender vectors with corresponding indexes
def user2target(users, targets):
    target_values = []
    for user in users:
        target_values.append(targets[user])
    return target_values






#########################################################################################################################
# Changes tokenized words to their corresponding ids in vocabulary
#
# input: list (tweets) - List of tweets
#        dict (vocab)  - Dictionary of the vocabulary of GloVe
#
# output: list (batch_tweet_ids) - List of corresponding ids of words in the tweet w.r.t. vocabulary
def word2id(tweets, vocab):
    user_batch = []

    for i in range(FLAGS.batch_size): #loop of users
        batch_tweet_ids = []

        for tweet in tweets[i]: #loop of tweets
            tweet_ids = []
            for word in tweet: #loop in words of tweet
                if word != "PAD":
                    word = word.lower()

                try:
                    tweet_ids.append(vocab[word])
                except:
                    tweet_ids.append(vocab["UNK"])

            batch_tweet_ids.append(tweet_ids)

        user_batch.append(batch_tweet_ids)

    return user_batch



#########################################################################################################################
# Changes tokenized words to their corresponding ids in vocabulary
# Works on user level batch system
#
# input: list (tweets) - List of tweets
#        dict (vocab)  - Dictionary of the vocabulary of char embeddings
#
# output: list (batch_tweet_ids) - List of corresponding ids of words in the tweet w.r.t. vocabulary
def char2id(tweets, char_list):
    batch_tweet_ids = []
    for user in tweets:
        user_tweets = []
        for tweet in user:
            tweet_ids = []
            for word in tweet:
                for char in word:
                    if char != 'P':
                        char = char.lower()
                    try:
                        tweet_ids.append(char_list[ord(char)])
                    except:
                        tweet_ids.append(char_list[ord('U')])
            user_tweets.append(tweet_ids)
        batch_tweet_ids.append(user_tweets)
    return batch_tweet_ids





#########################################################################################################################
# Prepares batch data, also adds padding to tweets
#
# input: list (tweets)  - List of tweets corresponding to the authors in:
#	     list (users)   - Owner of the tweets
#	     dict (targets) - Ground-truth gender vector of each owner
#	     list (seq_len) - Sequence length for tweets
#	     int  (iter_no) - Current # of iteration we are on
#
# output: list (batch_input)       - Ids of each words to be used in tf_embedding_lookup
# 	      list (batch_output)      - Target values to be fed to the rnn
#	      list (batch_sequencelen) - Number of words in each tweet(gives us the # of time unrolls)
def prepWordBatchData_tweet(tweets, users, targets, seq_len, iter_no):
    start = iter_no * FLAGS.batch_size
    end = iter_no * FLAGS.batch_size + FLAGS.batch_size

    if end > len(tweets):
        end = len(tweets)

    batch_tweets = tweets[start:end]
    batch_users = users[start:end]
    batch_sequencelen = seq_len[start:end]

    batch_output_temp = user2target(batch_users, targets)

    # prepare input by adding padding
    tweet_lengths = [len(tweet) for tweet in batch_tweets]
    max_tweet_length = max(tweet_lengths)

    batch_input = []
    for i in range(FLAGS.batch_size):
        tweet = batch_tweets[i]
        padded_tweet = []
        for j in range(max_tweet_length):
            if len(tweet) > j:
                padded_tweet.append(tweet[j])
            else:
                padded_tweet.append("PAD")
        batch_input.append(padded_tweet)


    c = list(zip(batch_input, batch_output_temp, batch_sequencelen))
    random.shuffle(c)
    tweet_batch, targets_batch, seqlens_batch = zip(*c)

    return tweet_batch, targets_batch, seqlens_batch



#########################################################################################################################
# Prepares batch data, also adds padding to tweets
#
# input: list (tweets)  - List of tweets corresponding to the authors in:
#	     list (users)   - Owner of the tweets
#	     dict (targets) - Ground-truth gender vector of each owner
#	     list (seq_len) - Sequence length for tweets
#	     int  (iter_no) - Current # of iteration we are on
#
# output: list (batch_input)       - Ids of each words to be used in tf_embedding_lookup
# 	      list (batch_output)      - Target values to be fed to the rnn
#	      list (batch_sequencelen) - Number of words in each tweet(gives us the # of time unrolls)
def prepWordBatchData(tweets, users, targets, seq_len, iter_no):
	numof_total_tweet = FLAGS.batch_size * FLAGS.tweet_per_user

	start = iter_no * numof_total_tweet
	end = iter_no * numof_total_tweet + numof_total_tweet

	if end > len(tweets):
		end = len(tweets)

	batch_tweets = tweets[start:end]
	batch_users = users[start:end]
	batch_sequencelen = seq_len[start:end]

	batch_targets = user2target(batch_users, targets)

	# prepare input by adding padding
	tweet_lengths = [len(tweet) for tweet in batch_tweets]
	max_tweet_length = max(tweet_lengths)

	batch_input = []
	for i in range(numof_total_tweet):
		tweet = batch_tweets[i]
		padded_tweet = []
		for j in range(max_tweet_length):
			if len(tweet) > j:
				padded_tweet.append(tweet[j])
			else:
				padded_tweet.append("PAD")
		batch_input.append(padded_tweet)


	#reshape the input for shuffling operation
	tweet_batches = np.reshape(np.asarray(batch_input), (FLAGS.batch_size, FLAGS.tweet_per_user, max_tweet_length)).tolist()
	target_batches = np.reshape(np.asarray(batch_targets), (FLAGS.batch_size, FLAGS.tweet_per_user, 2)).tolist()
	seqlen_batches = np.reshape(np.asarray(batch_sequencelen), (FLAGS.batch_size, FLAGS.tweet_per_user)).tolist()

	#prepare the target values
	target_values = []
	for i in range(len(target_batches)):
		target_values.append(target_batches[i][0]) 
	target_batches = np.reshape(np.asarray(target_values), (FLAGS.batch_size, 2)).tolist()

	'''
	#user level shuffling
	c = list(zip(tweet_batches, target_batches, seqlen_batches))
	random.shuffle(c)
	tweet_batches, target_batches, seqlen_batches = zip(*c)
	'''

	tweet_batches = list(tweet_batches)
	target_values = list(target_values)
	seqlen_batches = list(seqlen_batches)

	#tweet level shuffling
	for i in range(FLAGS.batch_size):
		c = list(zip(tweet_batches[i], seqlen_batches[i]))
		random.shuffle(c)
		tweet_batches[i], seqlen_batches[i] = zip(*c)

	tweet_batches = list(tweet_batches)
	seqlen_batches = list(seqlen_batches)

	return tweet_batches, target_batches, seqlen_batches








def prepCharBatchData(tweets, users, targets, iter_no):
    numof_total_tweet = FLAGS.batch_size * FLAGS.tweet_per_user

    start = iter_no * numof_total_tweet
    end = iter_no * numof_total_tweet + numof_total_tweet

    if end > len(tweets):
        end = len(tweets)

    batch_tweets = tweets[start:end]
    batch_users = users[start:end]

    batch_output = user2target(batch_users, targets)

    batch_input = list()

    for tweet in batch_tweets:
        tweet_char_list = list()
        for word in tweet:
            tweet_char_list.extend([char for char in word.lower()])

        size = len(tweet_char_list)
        if size < FLAGS.sequence_length:
            for i in range(FLAGS.sequence_length - size):
                tweet_char_list.append('P')

        batch_input.append(tweet_char_list)

    # reshape the input for shuffling operation
    tweet_batches = np.reshape(np.asarray(batch_input),(FLAGS.batch_size, FLAGS.tweet_per_user, FLAGS.sequence_length)).tolist()
    target_batches = np.reshape(np.asarray(batch_output), (FLAGS.batch_size, FLAGS.tweet_per_user, 2)).tolist()

    # prepare the target values
    target_values = []
    for i in range(len(target_batches)):
        target_values.append(target_batches[i][0])
    target_batches = np.reshape(np.asarray(target_values), (FLAGS.batch_size, 2)).tolist()

    '''
    # user level shuffling
    c = list(zip(tweet_batches, target_batches))
    random.shuffle(c)
    tweet_batches, target_batches = zip(*c)
    '''

    tweet_batches = list(tweet_batches)

    # tweet level shuffling
    for i in range(FLAGS.batch_size):
        c = list(zip(tweet_batches[i]))
        random.shuffle(c)
        tweet_batches[i] = zip(*c)

    tweet_batches = list(tweet_batches)
    target_batches = list(target_batches)

    return tweet_batches, target_batches





#########################################################################################################################
# partites the data into 3 part training, validation, test
#
# input: list (tweets)  - List of tweets corresponding to the authors in:
#	     list (users)   - Owner of the tweets
#	     list (seq_len) - Sequence length for tweets
#
# output: output_format : usagetype_datatype
#         list ("usagetype"_tweets)       - Group of tweets partitioned according to the FLAGS."usagetype"_set_size
# 	      list ("usagetype"_users)        - Group of users partitioned according to the FLAGS."usagetype"_set_size
#	      list ("usagetype"_seqlengths)   - Group of seqlengths partitioned according to the FLAGS."usagetype"_set_size
def partite_dataset(tweets, users, seq_lengths):

    training_set_size = int(len(tweets) * FLAGS.training_set_size)
    valid_set_size = int(len(tweets) * FLAGS.validation_set_size) + training_set_size

    training_tweets = tweets[:training_set_size]
    valid_tweets = tweets[training_set_size:valid_set_size]
    test_tweets = tweets[valid_set_size:]

    training_users = users[:training_set_size]
    valid_users = users[training_set_size:valid_set_size]
    test_users = users[valid_set_size:]

    training_seq_lengths = seq_lengths[:training_set_size]
    valid_seq_lengths = seq_lengths[training_set_size:valid_set_size]
    test_seq_lengths = seq_lengths[valid_set_size:]

    print("\ttraining set size=" + str(len(training_tweets)) + " validation set size=" + str(len(valid_tweets)) + " test set size=" + str(len(test_tweets)))

    return training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, test_tweets, test_users, test_seq_lengths





#########################################################################################################################
# Shuffles the data and partites it into 3 part training, validation, test
#
# input: list (tweets)  - List of tweets corresponding to the authors in:
#	     list (users)   - Owner of the tweets
#	     list (seq_len) - Sequence length for tweets
#
# output: output_format : usagetype_datatype
#         list ("usagetype"_tweets)       - Group of tweets partitioned according to the FLAGS."usagetype"_set_size
# 	      list ("usagetype"_users)        - Group of users partitioned according to the FLAGS."usagetype"_set_size
#	      list ("usagetype"_seqlengths)   - Group of seqlengths partitioned according to the FLAGS."usagetype"_set_size
def partite_dataset_tweet(tweets, users, seq_lengths):

	c = list(zip(tweets, users, seq_lengths))
	random.shuffle(c)
	tweets, users, seq_lengths = zip(*c)

	training_set_size = int(len(tweets) * FLAGS.training_set_size)
	valid_set_size = int(len(tweets) * FLAGS.validation_set_size) + training_set_size

	training_tweets = tweets[:training_set_size]
	valid_tweets = tweets[training_set_size:valid_set_size]
	test_tweets = tweets[valid_set_size:]

	training_users = users[:training_set_size]
	valid_users = users[training_set_size:valid_set_size]
	test_users = users[valid_set_size:]

	training_seq_lengths = seq_lengths[:training_set_size]
	valid_seq_lengths = seq_lengths[training_set_size:valid_set_size]
	test_seq_lengths = seq_lengths[valid_set_size:]

	print("\ttraining set size=" + str(len(training_tweets)) + " validation set size=" + str(len(valid_tweets)) + " test set size=" + str(len(test_tweets)))

	return training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, test_tweets, test_users, test_seq_lengths
