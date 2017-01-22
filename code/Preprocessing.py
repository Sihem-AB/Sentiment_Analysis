#!/usr/bin/python3


"""

Requirements:
	This file requires to download "punkt" model.
	This file requires to download "stopwords" corpora.
	==> Download them by typing nlkt.download()

"""


import Utils
import Preprocessing
import TermFrequencyProcessing
import FeatureSelection
import FileToReview

import string
import re
import os
import sys
import json


from nltk.data import load
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.sentiment.util import mark_negation


class Preprocessing(object):
	def __init__(self, pos_path, neg_path, selected_DB, pos_reviews=None, neg_reviews=None, is_bigrams=False):
		self.pos_path = pos_path 
		self.neg_path = neg_path
		self.pos_reviews = pos_reviews
		self.neg_reviews = neg_reviews
		self.selected_DB = selected_DB
		self.is_bigrams = is_bigrams

		self.V = {}   # vocabs
		self.STOPWORDS = stopwords.words('english')
		self.pos_json_filename = "pos_vocab.json"
		self.neg_json_filename = "neg_vocab.json"

		self.nb_pos_review = None
		self.nb_neg_review = None


#############################################
# Getter/Setter
#############################################

	def get_v(self):
		return self.V

	def get_pos_path(self):
		return self.pos_path

	def get_neg_path(self):
		return self.neg_path

	def get_pos_reviews(self):
		return self.pos_reviews

	def get_neg_reviews(self):
		return self.neg_reviews

	def get_nb_pos_review(self):
		return self.nb_pos_review

	def get_nb_neg_review(self):
		return self.nb_neg_review

	
	def set_v(self, V):
		self.V = V

	def set_pos_path(self, path):
		self.pos_path = path
	
	def set_neg_path(self, path):
		self.neg_path = path

	def set_pos_reviews(self, pos_reviews):
		self.pos_reviews = pos_reviews

	def set_neg_reviews(self, neg_reviews):
		self.neg_reviews = neg_reviews

	def set_nb_pos_review(self, nb_pos_review):
		self.nb_pos_review = nb_pos_review

	def set_nb_neg_review(self, nb_neg_review):
		self.nb_neg_review = nb_neg_review




#############################################
# Read/Write
#############################################

	"""
	An example of structured data from positive reviews, so-called pos_vocabs

	{	 
		"sentiment_class": 1, 		 ===============================> POSITIVE CLASS
		"nb_word": 69, 
		"reviews": [
			{
				"rating": 7,
				"nb_sentence": 3, 
				"nb_word": 35, 
				"sentences": [
					{"first": 3, "care": 1, "human": 1, "love": 1, "taken": 1, "one": 1},
					{"scene": 1, "cannaval": 1, "believ": 1, "steal": 1, "everi": 1, "bobbi": 1}, 
					{"lead": 1, "screen": 1, "1940": 1, "look": 1, "presenc": 1, "man": 1, "'s": 1} 
				]
			},
			{
				"rating": 6,
				"nb_sentence": 3, 
				"nb_word": 34, 
				"sentences": [
					{"first": 3, "care": 1, "human": 1, "love": 1, "taken": 1, "one": 1},
					{"scene": 1, "cannaval": 1, "believ": 1, "steal": 1, "everi": 1, "bobbi": 1}, 
					{"lead": 1, "screen": 1, "1940": 1, "look": 1, "presenc": 1, "man": 1, "'s": 1} 
				]
			}
		]
	}

	"""


	"""
		Read from the json files and get preprocessed vocabulary
	"""
	def read_vocab(self):	
		pos_path = None
		neg_path = None

		if self.selected_DB == Utils.DB_ONE:
			pos_path = Utils.get_parent_directory_for_file(self.pos_path)
			neg_path = Utils.get_parent_directory_for_file(self.neg_path)
		elif self.selected_DB == Utils.DB_TWO:
			pos_path = self.pos_path
			neg_path = self.neg_path

		with open(pos_path + "/" + self.pos_json_filename, 'r') as json_file:
			json_text = json_file.read()
			pos_vocabs = json.loads(json_text)

		with open(neg_path + "/" + self.neg_json_filename, 'r') as json_file:
			json_text = json_file.read()
			neg_vocabs = json.loads(json_text)

		V = {}
		V[Utils.POS] = pos_vocabs
		V[Utils.NEG] = neg_vocabs
		V["nb_word"] = pos_vocabs["nb_word"] + neg_vocabs["nb_word"]
		V["nb_review"] = pos_vocabs["nb_review"] + neg_vocabs["nb_review"]
		self.set_v(V)
		self.set_nb_pos_review( pos_vocabs["nb_review"] )
		self.set_nb_neg_review( neg_vocabs["nb_review"] )








	"""
		Write into 2 json files in order to save vocabulary and not to do preprocessing each time
	"""
	def write_vocab(self):
		pos_path = None
		neg_path = None

		if self.selected_DB == Utils.DB_ONE:
			pos_path = Utils.get_parent_directory_for_file(self.pos_path)
			neg_path = Utils.get_parent_directory_for_file(self.neg_path)
		elif self.selected_DB == Utils.DB_TWO:
			pos_path = self.pos_path
			neg_path = self.neg_path

		with open(pos_path + "/" + self.pos_json_filename, 'w') as json_file:
			json.dump(self.V[Utils.POS], json_file)

		with open(neg_path + "/" + self.neg_json_filename, 'w') as json_file:
			json.dump(self.V[Utils.NEG], json_file)



#############################################
# Other Methods
#############################################


	def extract_vocabulary(self):
		self._extract_vocabulary(self.pos_reviews, Utils.POS)
		self._extract_vocabulary(self.neg_reviews, Utils.NEG)
		V = self.get_v()
		V["nb_word"] = V[Utils.POS]["nb_word"] + V[Utils.NEG]["nb_word"]
		V["nb_review"] = V[Utils.POS]["nb_review"] + V[Utils.NEG]["nb_review"]
		self.set_v(V)
		self.set_nb_pos_review( V[Utils.POS]["nb_review"] )
		self.set_nb_neg_review( V[Utils.NEG]["nb_review"] )
		self.set_nb_neg_review( V[Utils.NEG]["nb_review"] )




	"""
		1) Extract each review from review matrix (i.e pos_reviews and neg_reviews)
		2) extract sentences from each review
		3) apply preprocessing operations on each extracted sentence
		4) find term frequency in each sentence
		5) append structured data into relevant vocab data (pos or neg)
		6) repeat steps from 1) through 5) for "neg" directory
	
	An example of the whole vocabs (both positive and negative)
	
	{
		"nb_word" : 46
		"1" : {	 
			"sentiment_class": 1, 		 ===============================> POSITIVE CLASS
			"nb_word": 28, 
			"reviews": [
				{
					"rating": 7,
					"nb_sentence": 2, 
					"nb_word": 19,		 ===============================> nb word before preprocessing 
					"sentences": [
						{"first": 3, "care": 1},
						{"scene": 1, "cannaval": 1, "believ": 1}, 
					]
				},
				{
					"rating": 9,
					"nb_sentence": 1, 
					"nb_word": 9, 
					"sentences": [
						{"first": 3, "care": 1}
					]
				}
			]
		},
		"-1" : {	 
			"sentiment_class": -1, 		 ===============================> NEGATIVE CLASS
			"nb_word": 18, 
			"reviews": [
				{
					"rating": 3,
					"nb_sentence": 2, 
					"nb_word": 10, 
					"sentences": [
						{"first": 1, "care": 1},
						{"scene": 1, "cannaval": 1, "believ": 1}, 
					]
				},
				{
					"rating": 2,
					"nb_sentence": 1, 
					"nb_word": 8, 
					"sentences": [
						{"first": 1, "care": 1}
					]
				}
			]
		}
	}
	"""	
	def _extract_vocabulary(self, aReviews, sent_class):
		self.V[sent_class] = {}
		self.V[sent_class]["nb_word"] = 0
		self.V[sent_class]["nb_review"] = 0
		self.V[sent_class]["sentiment_class"] = sent_class
		self.V[sent_class]["reviews"] = []


		"""
		an example of aReviews:

		array([['not bad :) ', '6'],
                ['Loved it', '9'],
                [' I can be pretty picky but loved it!', '9'],
                ['Enjoy enjoy the show!', '7']], 
                dtype='<U1')

		"""
		cpt = 0
		for sReview, rating in aReviews:
			self.V[sent_class]["nb_review"] += 1

			dReview = {}
			dReview["nb_word"] = 0
			dReview["nb_sentence"] = 0
			dReview["rating"] = int(rating)
			dReview["sentences"] = []
			dReview["sentences_ordered"] = []  # Will contain every word of the sentence in order
			dReview["id"] = cpt + len(aReviews)* sent_class
			cpt += 1
		
			sReview = self.clean_html(sReview)
			sentences = self.divide_into_sentences(sReview)
		
			for sentence in sentences:
				nb_word = self.count_words(sentence)
				aWords = self.sentence_preprocessing(sentence)
				dWords= self.find_term_frequency(aWords)
				dReview["nb_word"] += nb_word
				dReview["nb_sentence"] += 1 
				dReview["sentences"].append(dWords)
				dReview["sentences_ordered"].append(aWords)
				self.V[sent_class]["nb_word"] += nb_word
	
			self.V[sent_class]["reviews"].append(dReview)





	def sentence_preprocessing(self, sentence):
		sentence = self.lowercase(sentence)
		aSentence = self.tokenize(sentence)
		aSentence = self.handle_negation(aSentence)
		aSentence = self.remove_punctuation(aSentence)
		aSentence = self.remove_stopwords(aSentence)
		aSentence = self.apply_stemming(aSentence)

		if self.is_bigrams:
			aSentence = self.bigrams(aSentence)	

		return aSentence




	def find_term_frequency(self, aWords):
		dWords = {}
		for word in aWords:
			if word not in dWords:
				dWords[word] = 1
			else:
				dWords[word] += 1

		return dWords




	def count_words(self, sentence):
		return len(self.tokenize(sentence))





	def handle_negation(self, aWords):
		# http://www.nltk.org/_modules/nltk/sentiment/util.html#mark_negation
		return mark_negation(aWords)




	def lowercase(self, text):
		return text.lower()



	def apply_stemming(self, aWords):
		# uses Porter stemming
		stemmer = PorterStemmer()
		return [str(stemmer.stem(word)) for word in aWords] # output of stemmer.stem(term) is u'string




	def tokenize(self, sentence):	
		# returns a list of word tokens and also remove multiple spaces and \t, \n etc.
		return word_tokenize(sentence)



	"""
		input: ["I", "am", "happy"]
		Output: ["I am", "am happy"]
	"""
	def bigrams(self, aWords):
		bgs = bigrams(aWords) # nltk.bigrams returns a generator object
		return [t[0]+" "+t[1] for t in bgs]



	def remove_punctuation(self, aWords):
		return [x for x in aWords if not re.fullmatch('[' + string.punctuation + ']+', x)]
		# string.punctuation is "!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
		#it needs the "string" module to be imported



	def remove_stopwords(self, aWords):
		return [word for word in aWords if word not in self.STOPWORDS]




	"""
		source: http://www.nltk.org/api/nltk.tokenize.html

		This method needs the module "nltk.data" to be imported
		Return a list of sentences
	"""
	def divide_into_sentences(self, text):
		# sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		sent_detector = load('tokenizers/punkt/english.pickle')
		return sent_detector.tokenize(text.strip())



	def clean_html(self, html):
		"""
		Copied from NLTK package.
		Remove HTML markup from the given string.

		:param html: the HTML string to be cleaned
		"""

		# First we remove inline JavaScript/CSS:
		cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
		# Then we remove html comments. This has to be done before removing regular
		# tags since comments can contain '>' characters.
		cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
		# Next we can remove the remaining tags:
		cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
		# Finally, we deal with whitespace
		cleaned = re.sub(r"&nbsp;", " ", cleaned)	
		cleaned = re.sub(r"  ", " ", cleaned)
		cleaned = re.sub(r"  ", " ", cleaned)
		
		return cleaned.strip()


"""
Input :
    pos_path : repertory positive directory
    neg_path : repertory negative directory
    selected_DB : which data we use ? imdb or the first one ? : Utils.DB_TWO or Utilis.DB_ONE
    is_bigrams : Do we use bigrams to compute tfidf etc ?
    k : percentage of most import word we keep (multual information)
    method : Mutual information or other method
    feature_space : if the feature has already been computed on an other dataset
Output :
    return the review preprocessed and with reduced vocabulary
    return fs : FeatureSelection Object
    return feature_space : if mutual information is choosen, feature_space contain the words we keep (only if feature_space is not given as input)
"""
def do_preprocessing(pos_path, neg_path, selected_DB, is_bigrams, k=None, method=None, features_space=None):
	f2r = FileToReview.FileToReview(pos_path, neg_path, selected_DB)
	pos_reviews, neg_reviews = f2r.buildReviewMatrix()

	# get a new instance for preprocessing
	# The new instance needs to know where positive and negative review directories are, also database no
	prep = Preprocessing(pos_path, neg_path, selected_DB, pos_reviews, neg_reviews, is_bigrams)

	# extract positive and negative vocabularies
	prep.extract_vocabulary()
	# print extracted vocabularies in dictionnary (json) format
	vocabs = prep.get_v()

	nb_neg_review = prep.get_nb_neg_review()
	nb_pos_review = prep.get_nb_pos_review()


    # get a new instance
    # The new instance needs to know where positive and negative review directories are, also database no
	tfp = TermFrequencyProcessing.TermFrequencyProcessing(pos_path, neg_path, selected_DB)
	tfp.compute_terms_frequency(vocabs)
	# print(tfp.get_overall_terms_frequency())
	# print(tfp.get_reviews_info())
	T = tfp.get_overall_terms_frequency()


	fs = FeatureSelection.FeatureSelection(T, nb_neg_review, nb_pos_review)

	if not features_space:
		features_space = fs.build_features_space(k, method)
		reduced_vocabs = fs.reduce_vocabs(vocabs, features_space)

		return vocabs, reduced_vocabs, fs, features_space

	reduced_vocabs = fs.reduce_vocabs(vocabs, features_space)
	return vocabs, reduced_vocabs, fs