#!/usr/bin/python3

import Preprocessing
import Utils
import json

class TermFrequencyProcessing(object):
	"""
		Input:
			pos_path: path to positive reviews
			neg_path: path to negative reviews
			selected_DB: Utils.DB_ONE or Utils.DB_TWO
	"""
	def __init__(self, pos_path, neg_path, selected_DB):
		self.pos_path = pos_path 
		self.neg_path = neg_path
		self.selected_DB = selected_DB
		self.T = {}   # overal term frequency
		self.pos_terms_json_filename = "pos_terms_freq.json"
		self.neg_terms_json_filename = "neg_terms_freq.json"
		
		# saving reviews info into list
		self.reviews_info = {}
		self.reviews_info[Utils.NEG] = []
		self.reviews_info[Utils.POS] = []
		self.nb_neg_review = 0
		self.nb_pos_review = 0
		self.nb_word_in_neg_reviews = 0
		self.nb_word_in_pos_reviews = 0
		self.pos_reviews_info_filename = "pos_reviews_info.json"
		self.neg_reviews_info_filename = "neg_reviews_info.json"


#############################################
# Getter/Setter
#############################################

	def get_overall_terms_frequency(self):
		return self.T

	def get_pos_path(self):
		return self.pos_path
	
	def get_neg_path(self):
		return self.neg_path

	def get_reviews_info(self):
		return self.reviews_info
	
	def get_nb_pos_review(self):
		return self.nb_pos_review

	def get_nb_neg_review(self):
		return self.nb_neg_review

	def get_nb_word_in_neg_reviews(self):
		return self.nb_word_in_neg_reviews
	
	def get_nb_word_in_pos_reviews(self):
		return self.nb_word_in_pos_reviews


	def set_overall_terms_frequency(self, T):
		self.T = T

	def set_pos_path(self, path):
		self.pos_path = path
	
	def set_neg_path(self, path):
		self.neg_path = path

	def set_reviews_info(self, reviews_info):
		self.reviews_info = reviews_info

	def set_nb_pos_review(self, nb_pos_review):
		self.nb_pos_review = nb_pos_review

	def set_nb_neg_review(self, nb_neg_review):
		self.nb_neg_review = nb_neg_review

	def set_nb_word_in_neg_reviews(self, nb_word_in_neg_reviews):
		self.nb_word_in_neg_reviews = nb_word_in_neg_reviews 
	
	def set_nb_word_in_pos_reviews(self, nb_word_in_pos_reviews):
		self.nb_word_in_pos_reviews = nb_word_in_pos_reviews 


#####################################################################################################
# Read/Write
#####################################################################################################

##### Save review information (nb of positive/negative review, review id, nb of word in review and ratings)

	"""
	An example of self.reviews_info:

	{
		-1 : [			===============> negative reviews
			(3, 42),			===============> first element is review id, 2nd is nb word
			(2, 85),
			(4, 40)
		],
		1 : [			===============> positive reviews
			(8, 16),
			(7, 78)
		]
	}

	"""

	def read_reviews_info(self):
		pos_path = None
		neg_path = None

		if self.selected_DB == Utils.DB_ONE:
			pos_path = Utils.get_parent_directory_for_file(self.pos_path)
			neg_path = Utils.get_parent_directory_for_file(self.neg_path)
		elif self.selected_DB == Utils.DB_TWO:
			pos_path = self.pos_path
			neg_path = self.neg_path		

		with open(pos_path + "/" + self.pos_reviews_info_filename, 'r') as json_file:
			json_text = json_file.read()
			pos_reviews_info = json.loads(json_text)

		with open(neg_path + "/" + self.neg_reviews_info_filename, 'r') as json_file:
			json_text = json_file.read()
			neg_reviews_info = json.loads(json_text)
		
		self.reviews_info = {}
		self.reviews_info[Utils.NEG] = neg_reviews_info["reviews_info"] 
		self.reviews_info[Utils.POS] = pos_reviews_info["reviews_info"]
		# update also self.nb_neg_review and self.nb_pos_review
		self.nb_neg_review = neg_reviews_info["nb_review"]
		self.nb_pos_review = pos_reviews_info["nb_review"]

		total = 0
		for review in neg_reviews_info["reviews_info"]:
			total += review[1]
		self.nb_word_in_neg_reviews = total

		total = 0
		for review in pos_reviews_info["reviews_info"]:
			total += review[1]
		self.nb_word_in_pos_reviews = total



	def write_reviews_info(self):	
		pos_path = None
		neg_path = None

		if self.selected_DB == Utils.DB_ONE:
			pos_path = Utils.get_parent_directory_for_file(self.pos_path)
			neg_path = Utils.get_parent_directory_for_file(self.neg_path)
		elif self.selected_DB == Utils.DB_TWO:
			pos_path = self.pos_path
			neg_path = self.neg_path		


		with open(pos_path + "/" + self.pos_reviews_info_filename, 'w') as json_file:
			pos_reviews_info = {}
			pos_reviews_info["nb_review"] = self.nb_pos_review
			# extract only terms frequency for positive reviews
			pos_reviews_info["reviews_info"] = self.reviews_info[Utils.POS]
			json.dump(pos_reviews_info, json_file)

		with open(neg_path + "/" + self.neg_reviews_info_filename, 'w') as json_file:
			neg_reviews_info = {}
			neg_reviews_info["nb_review"] = self.nb_neg_review
			# extract only terms frequency for negative reviews
			neg_reviews_info["reviews_info"] = self.reviews_info[Utils.NEG]
			json.dump(neg_reviews_info, json_file)




#####################################################################################################
#### save terms frequency

	"""

		An example of neg_terms_json_file: 
			{"soon": {"nb_review": 1, "reviews": [[3, 1]]}, "predecessor": {"nb_review": 1, "reviews": [[2, 3]]}, ...}

		An example of pos_terms_json_file: 
			{"soon": {"nb_review": 2, "reviews": [[1, 1], [2, 1]]}, ...}


		overall_terms_frequency will be:
		{
			"soon": {
				-1: {				===============> negative review
					"nb_review": 1,
					"reviews": [
						[3, 1]			===========> the word "soon" is in 1 negative review whose the id is 3 and the term frequency is 1
					]
				},
				1: {				===============> positive review
					"nb_review": 2,
					"reviews": [
						[1, 1],
						[2, 1]
					]
				}
			},
			"predecessor": {
				-1: {
					"nb_review": 1,
					"reviews": [
						[2, 3]
					]
				}
			},
			.
			.
			.
		}

	"""



	"""
		Read from the json files and get preprocessed vocabulary
	"""
	def read_terms_frequency(self):	
		pos_path = None
		neg_path = None

		if self.selected_DB == Utils.DB_ONE:
			pos_path = Utils.get_parent_directory_for_file(self.pos_path)
			neg_path = Utils.get_parent_directory_for_file(self.neg_path)
		elif self.selected_DB == Utils.DB_TWO:
			pos_path = self.pos_path
			neg_path = self.neg_path		

		with open(pos_path + "/" + self.pos_terms_json_filename, 'r') as json_file:
			json_text = json_file.read()
			pos_terms = json.loads(json_text)

		with open(neg_path + "/" + self.neg_terms_json_filename, 'r') as json_file:
			json_text = json_file.read()
			neg_terms = json.loads(json_text)


		# reset overall term frequency
		self.T.clear() # TODO is it usefull?


		# first read terms and their reviews and frequency information corresponding to negative reviews
		for term, tf_val in neg_terms.items():
			self.T[term] = {}
			self.T[term][Utils.NEG] = tf_val

		# first read terms and their reviews and frequency information corresponding to positive reviews
		for term, tf_val in pos_terms.items():
			if term not in self.T.keys():
				self.T[term] = {}
			self.T[term][Utils.POS] = tf_val
					
	


	"""
		Write into 2 json files in order to save vocabulary and not to do preprocessing each time
	"""
	def write_terms_frequency(self):
		pos_path = None
		neg_path = None

		if self.selected_DB == Utils.DB_ONE:
			pos_path = Utils.get_parent_directory_for_file(self.pos_path)
			neg_path = Utils.get_parent_directory_for_file(self.neg_path)
		elif self.selected_DB == Utils.DB_TWO:
			pos_path = self.pos_path
			neg_path = self.neg_path		

		with open(pos_path + "/" + self.pos_terms_json_filename, 'w') as json_file:
			# extract only terms frequency for positive reviews
			pos_T = {key: val[Utils.POS] for key, val in self.T.items() if Utils.POS in val.keys()} 
			json.dump(pos_T, json_file)

		with open(neg_path + "/" + self.neg_terms_json_filename, 'w') as json_file:
			# extract only terms frequency for negative reviews
			neg_T = {key: val[Utils.NEG] for key, val in self.T.items() if Utils.NEG in val.keys()}
			json.dump(neg_T, json_file)



#####################################################################################################
# term frequency methods
#####################################################################################################

	def compute_terms_frequency(self, V):
		if not V:
			print("There is no vocabulary. Can not compute terms frequency")
			return -1

		self.T.clear() # TODO is it useful?
		
		for sentiment_class in [Utils.NEG, Utils.POS]:
			vocabs = V[sentiment_class]
			# _compute_terms_frequency() updates self.T
			self._compute_terms_frequency(vocabs, sentiment_class)


	def _compute_terms_frequency(self, vocabs, sentiment_class):
		reviews = vocabs["reviews"]

		for review in reviews:
			rating = review["rating"]
			nb_word = review["nb_word"]
			review_id = self.update_reviews_info(rating, nb_word, sentiment_class)

			review_terms = self.merge_terms_frequency_in_review(review)	

			# update_overall_terms_frequency() updates self.T at each call
			self.update_overall_terms_frequency(review_terms, review_id, sentiment_class)



	"""
	Update reviews_info list.

	Return a new review id

	An example:
	
	{
		-1 : [
			(3, 42),
			(2, 85),
			(4, 40)
		],
		1 : [
			(8, 16),
			(7, 78)
		]
	}
	
	reviews_info is made up reviews id for both negative and positive reviews.
	For instance, for negative reviews, there are 3 reviews.
	1st review has 3 rating and contains 42 words.
	id of the 1st review is 0.
	id of the 2nd review is 1.
	id of the 3rd review is 2.

	"""
	def update_reviews_info(self, rating, nb_word, sentiment_class):
		self.reviews_info[sentiment_class].append( (rating, nb_word) )

		new_id = None
		if sentiment_class == Utils.NEG:
			cur_neg_nb_word = self.get_nb_word_in_neg_reviews()
			self.set_nb_word_in_neg_reviews(cur_neg_nb_word + nb_word)

			new_id = self.get_nb_neg_review()
			self.set_nb_neg_review(new_id + 1)
		elif sentiment_class == Utils.POS:
			cur_pos_nb_word = self.get_nb_word_in_pos_reviews()
			self.set_nb_word_in_pos_reviews(cur_pos_nb_word + nb_word)

			new_id = self.get_nb_pos_review()
			self.set_nb_pos_review(new_id + 1)

		return new_id




	"""

	An example of output for both negative and positive reviews

	{
		"term1": {
			"-1": {			==============> negative review (sentiment class)
				"nb_review": nb,
				"reviews": [
					(review id, term frequency),
					(review id, term frequency)
				]
			},
			"1": {
				"nb_review": nb,
				"reviews": [
					(review id, term frequency),
					(review id, term frequency)
				]
			}
		},
		"term2": {
			"-1": {
				.
				.
			},
			"1": {
				.
				.
			}
		}
	}
	"""
	def update_overall_terms_frequency(self, review_terms, review_id, sentiment_class):
		T = self.get_overall_terms_frequency()
		
		# review_terms contains all the word in a review  whose its id is review_id
		for term, freq in review_terms.items():
			if term not in T.keys():
				T[term] = {}

			if sentiment_class in T[term].keys():
				T[term][sentiment_class]["nb_review"] += 1
				T[term][sentiment_class]["reviews"].append( (review_id, freq) )
			elif sentiment_class not in T[term].keys():
				T[term][sentiment_class] = {
						"nb_review" :  1,
						"reviews" : [
							(review_id, freq)
						]
					}

		self.set_overall_terms_frequency(T)






	"""
		Input: review, i.e. a list of sentence

		Each sentence is a list of tokens(i.e terms or words).
		The term frequency is associated with the terms.
		It is possible that such a word, let's say "hello", is both in 1st sentence and 2nd sentence.
		In this case, it needs to sum up the frequencies of the word "hello"
		 to use term frequency in review level instead of sentence level.
		This method computes the terms frequency in review level


		An example of output:

		terms: {
				"good" : freq,
				"bad" : freq
			}
		
	"""
	def merge_terms_frequency_in_review(self, review):
		terms = {}
		sentences = review["sentences"]

		for sentence in sentences:
			for word, freq in sentence.items():
				if word in terms.keys():
					terms[word] += freq
				else:
					terms[word] = freq

		return terms


