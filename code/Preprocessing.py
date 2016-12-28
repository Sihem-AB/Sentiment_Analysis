#!/usr/bin/python3


"""

Requirements:
	This file requires to download "punkt" model.
	This file requires to download "stopwords" corpora.
	==> Download them by typing nlkt.download()

"""


import Utils

import string
import re
import os
import sys
import json


from nltk.data import load
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords


class Preprocessing(object):
	def __init__(self, pos_path, neg_path, selected_DB):
		self.pos_path = pos_path 
		self.neg_path = neg_path
		self.selected_DB = selected_DB
		self.V = {}   # vocabs
		self.STOPWORDS = stopwords.words('english')
		self.pos_json_filename = "pos_vocab.json"
		self.neg_json_filename = "neg_vocab.json"


#############################################
# Getter/Setter
#############################################

	def get_v(self):
		return self.V


	def get_pos_path(self):
		return self.pos_path

	
	def get_neg_path(self):
		return self.neg_path

	
	def set_v(self, V):
		self.V = V


	def set_pos_path(self, path):
		self.pos_path = path

	
	def get_neg_path(self, path):
		self.neg_path = path





#############################################
# Read/Write
#############################################

	"""
	An example of structured data from positive reviews, so-called pos_vocabs

	{	 
		"sentiment_class": 1, 		 ===============================> POSITIVE CLASS
		"nb_word": 42, 
		"reviews": [
			{
				"nb_sentence": 5, 
				"nb_word": 21, 
				"sentences": [
					{"first": 3, "care": 1, "human": 1, "love": 1, "taken": 1, "one": 1},
					{"scene": 1, "cannaval": 1, "believ": 1, "steal": 1, "everi": 1, "bobbi": 1}, 
					{"lead": 1, "screen": 1, "1940": 1, "look": 1, "presenc": 1, "man": 1, "'s": 1} 
				]
			},
			{
				"nb_sentence": 5, 
				"nb_word": 21, 
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
		self.set_v(V)




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
		if self.selected_DB == Utils.DB_ONE:
			self.extract_vocab_DB_one()
		elif self.selected_DB == Utils.DB_TWO:
			self.extract_vocab_DB_two()



	def extract_vocab_DB_one(self):
		self._extract_vocab_DB_one(self.pos_path, Utils.POS)
		self._extract_vocab_DB_one(self.neg_path, Utils.NEG)
		V = self.get_v()
		V["nb_word"] = V[Utils.POS]["nb_word"] + V[Utils.NEG]["nb_word"]
		self.set_v(V)


	def extract_vocab_DB_two(self):
		self._extract_vocab_DB_two(self.pos_path, Utils.POS)
		self._extract_vocab_DB_two(self.neg_path, Utils.NEG)
		V = self.get_v()
		V["nb_word"] = V[Utils.POS]["nb_word"] + V[Utils.NEG]["nb_word"]
		self.set_v(V)



	def _extract_vocab_DB_one(self, path, sent_class):
		if not( Utils.is_file(path) ):
			print("error: path is not an existing file")
			# TODO raise an error
			return 0


		self.V[sent_class] = {}
		self.V[sent_class]["nb_word"] = 0
		self.V[sent_class]["sentiment_class"] = sent_class
		self.V[sent_class]["reviews"] = []


		with open("./" + path, "r") as f:
			for sReview in f:
				dReview = {}
				dReview["nb_word"] = 0
				dReview["nb_sentence"] = 0
				dReview["rating"] = Utils.POS_RATING_DEFAULT if sent_class == Utils.POS else Utils.NEG_RATING_DEFAULT
				dReview["sentences"] = []
				
				sReview = self.clean_html(sReview)
				sentences = self.divide_into_sentences(sReview)
				
				for sentence in sentences:
					aWords = self.sentence_preprocessing(sentence)
					dWords, nb_word = self.find_term_frequency(aWords)
					dReview["nb_word"] += nb_word
					dReview["nb_sentence"] += 1 
					dReview["sentences"].append(dWords)
					self.V[sent_class]["nb_word"] += nb_word
			
				self.V[sent_class]["reviews"].append(dReview)





	"""
		1) Read each review in "pos" directory
		2) extract sentences from each review
		3) apply preprocessing operations on each extracted sentence
		4) find term frequency in each sentence
		5) append structured data into relevant vocab data (pos or neg)
		6) repeat steps from 1) through 5) for "neg" directory
	
	An example of the whole vocabs (both positive and negative)
	
	{
		"nb_word" : 18
		"1" : {	 
		"sentiment_class": 1, 		 ===============================> POSITIVE CLASS
		"nb_word": 11, 
		"reviews": [
			{
				"nb_sentence": 2, 
				"nb_word": 7, 
				"sentences": [
					{"first": 3, "care": 1, ...},
					{"scene": 1, "cannaval": 1, "believ": 1, ...}, 
				]
			},
			{
				"nb_sentence": 1, 
				"nb_word": 4, 
				"sentences": [
					{"first": 3, "care": 1, ...}
				]
			}
		},
		"0" : {	 
		"sentiment_class": 0, 		 ===============================> NEGATIVE CLASS
		"nb_word": 7, 
		"reviews": [
			{
				"nb_sentence": 2, 
				"nb_word": 5, 
				"sentences": [
					{"first": 1, "care": 1, ...},
					{"scene": 1, "cannaval": 1, "believ": 1, ...}, 
				]
			},
			{
				"nb_sentence": 1, 
				"nb_word": 2, 
				"sentences": [
					{"first": 1, "care": 1, ...}
				]
			}
		}
	}
	"""	
	def _extract_vocab_DB_two(self, path, sent_class):
		if not( Utils.is_directory(path) ):
			print("error: path is not a directory")
			# TODO raise an error
			return 0


		self.V[sent_class] = {}
		self.V[sent_class]["nb_word"] = 0
		self.V[sent_class]["sentiment_class"] = sent_class
		self.V[sent_class]["reviews"] = []

		# get only .txt files and not .json files
		files = [f for f in os.listdir(path) if re.match(r'.*\.txt', f)]

		for filename in files:
			rating = self.extract_rating(filename)
			dReview = {}
			dReview["nb_word"] = 0
			dReview["nb_sentence"] = 0
			dReview["rating"] = rating
			dReview["sentences"] = []

			with open (path+"/"+filename, "r") as f:
				sReview = f.read()
				sReview = self.clean_html(sReview)
				sentences = self.divide_into_sentences(sReview)
				
				for sentence in sentences:
					aWords = self.sentence_preprocessing(sentence)
					dWords, nb_word = self.find_term_frequency(aWords)
					dReview["nb_word"] += nb_word
					dReview["nb_sentence"] += 1 
					dReview["sentences"].append(dWords)
					self.V[sent_class]["nb_word"] += nb_word
			
			self.V[sent_class]["reviews"].append(dReview)



	def sentence_preprocessing(self, sentence):
		sentence = self.lowercase(sentence)
		aSentence = self.tokenize(sentence)
		#aSentence = self.handle_negation(aSentence) # TODO
		aSentence = self.remove_punctuation(aSentence)
		aSentence = self.remove_stopwords(aSentence)
		aSentence = self.apply_stemming(aSentence)

		return aSentence




	def find_term_frequency(self, aWords):
		dWords = {}
		nb_term = 0
		for word in aWords:
			nb_term += 1

			if word not in dWords:
				dWords[word] = 1
			else:
				dWords[word] += 1

		return dWords, nb_term




	def extract_rating(self, filename):
		# a rating value could be 10 so 2 digit
		part = filename.split("_")[1]
		return part.split(".")[0] # rating




	def handle_negation(self, aWords):
		pass # TODO




	def lowercase(self, text):
		return text.lower()



	def apply_stemming(self, aWords):
		# uses Porter stemming
		stemmer = PorterStemmer()
		return [str(stemmer.stem(word)) for word in aWords] # output of stemmer.stem(term) is u'string




	def tokenize(self, sentence):	
		# returns a list of word tokens and also remove multiple spaces and \t, \n etc.
		return word_tokenize(sentence)




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
