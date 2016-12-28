#!/usr/bin/python3

import Utils
import TermFrequencyProcessing
from math import floor
from math import log


class FeatureSelection(object):
	def __init__(self, pos_path, neg_path, selected_DB, T, reviews_info, nb_neg_review, nb_pos_review, nb_word_in_neg_reviews, nb_word_in_pos_reviews):
		self.pos_path = pos_path 
		self.neg_path = neg_path
		self.selected_DB = selected_DB
		self.T = T   # overall terms frequency
		self.reviews_info = reviews_info
		self.nb_neg_review = nb_neg_review
		self.nb_pos_review = nb_pos_review
		self.nb_word_in_neg_reviews = nb_word_in_neg_reviews
		self.nb_word_in_pos_reviews = nb_word_in_pos_reviews


	"""
		An example of self.T:

        {
                "term1": {
                        "0": {                  ==============> negative review (sentiment class)
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
                        "0": {
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

###############################################################################
# Mutual Information
###############################################################################

	"""
		Input: 
			k : pourcentage of words with largest MI value. 
				It should be between 0 and 1. ==> Float
			
		Returns k% terms with largest values


		Compute feature utility.
        According to the source below, there are 3 feature utility function: mutual information, chi square test and frequency
        source: http://nlp.stanford.edu/IR-book/html/htmledition/feature-selection-1.html#fig:featselalg
        For now, there is only 1 available feature utility: Mutual Information (MI)
	"""
	def compute_MI(self, k=1):
		if k > 1 or k <= 0:
			k = 1 # default value

		L = self._compute_MI(self.nb_neg_review, self.nb_pos_review)
		card_L = len(L)
		# transform pourcentage into number
		top_k = floor(k * card_L) # i.e floor(30.2) = 30
		print("MI: The number of terms with largest values according to the parameter 'k' ====>", top_k, "\n")
		
		# sort list by mutual information value in descending order
		# extract top k terms
		return sorted(L, key=lambda x: x[1], reverse=True)[:top_k]

	

	def _compute_MI(self, nb_neg_review, nb_pos_review):
		"""
			conditional mutual information and dealing with zero probabilities. For this purpose, Utils.flexible_log() is created
			source: http://stats.stackexchange.com/questions/73502/conditional-mutual-information-and-how-to-deal-with-zero-probabilities
			
			formula source: http://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html


			N_11: number of positive review that contains the word
			N_10: number of negative review that contains the word
			N_01: number of positive review that doesn't contain the word
			N_00: number of negative review that doesn't contain the word

			N = N_11 + N_10 + N_01 + N_00
			N_1. = N_10 + N_11
			N_0. = N_00 + N_01
			N_.1 = N_01 + N_11
			N_.0 = N_10 + N_00

			formula:
				I(U;C) = (N_11/N * log_2(N*N_11/N_1.*N_.1) ) +
						 (N_01/N * log_2(N*N_01/N_0.*N_.1) ) +
						 (N_10/N * log_2(N*N_10/N_1.*N_.0) ) +
						 (N_00/N * log_2(N*N_00/N_0.*N_.0) )
		"""

		L = []
		for term, tf_val in self.T.items():
			N_11 = tf_val[Utils.POS]["nb_review"] if Utils.POS in tf_val.keys() else 0
			N_10 = tf_val[Utils.NEG]["nb_review"] if Utils.NEG in tf_val.keys() else 0
			N_01 = nb_pos_review - N_11
			N_00 = nb_neg_review - N_10
			N__1 = N_01 + N_11
			N_1_ = N_10 + N_11
			N__0 = N_10 + N_00
			N_0_ = N_00 + N_01
			N = N_11 + N_10 + N_01 + N_00

            # TODO break a line for the formula below
			result = ((N_11/N) * Utils.flexible_log( (N*N_11)/(N_1_*N__1) )) + ((N_01/N) * Utils.flexible_log( (N*N_01)/(N_0_*N__1) )) + ((N_10/N) * Utils.flexible_log( (N*N_10)/(N_1_*N__0) )) + ((N_00/N) * Utils.flexible_log( (N*N_00)/(N_0_*N__0) )) 

			L.append((term, result))

		return L


###############################################################################
# TF - IDF
###############################################################################


	def compute_TF_IDF(self, k=1):
		if k > 1 or k <= 0:
			k = 1 # default value

		L = self._compute_TF_IDF()
		card_L =len(L)
		# transform pourcentage into number
		top_k = floor(k * card_L)
		print("TF-IDF: The number of terms with largest values according to the parameter 'k' ====>", top_k, "\n")
		
		# sort list by tf-idf value in descending order
		# extract top k terms
		return sorted(L, key=lambda x: x[1], reverse=True)[:top_k]



	def _compute_TF_IDF(self):
		L = []

		for term, tf_info in self.T.items():
			idf = self.compute_IDF(term, tf_info)
			tf = self.compute_max_TF(term, tf_info)
			L.append( (term, tf*idf) )
	
		return L



	def compute_max_TF(self, term, tf_info):
		"""
			tf(w, d) = 1 + log(number of occurences of w in d)	
		"""

		pos_reviews_for_term = tf_info[Utils.POS]["reviews"] if Utils.POS in tf_info.keys() else []
		neg_reviews_for_term = tf_info[Utils.NEG]["reviews"] if Utils.NEG in tf_info.keys() else []
		

		"""
            "reviews": [
                  (review id, term frequency),
                  (review id, term frequency)
            ]
		"""

		# reviews contains review_id - term freq pairs.
		# so we can access to term frequency value associated with the term in each review
		# what we need is to get the max frequency value among all frequency values (if the term is frequent in a document d1 and is not frequent in d2, what is important to us is the frequency in d1)
		pos_freqs = list(map(lambda x: x[1], pos_reviews_for_term))
		neg_freqs = list(map(lambda x: x[1], neg_reviews_for_term))

		# it is possible that a word may occur just in one side (either negative or positive)
		# if a list object is empty, we can not perform max()
		max_pos_freq = max(pos_freqs) if len(pos_freqs) > 0 else 0
		max_neg_freq = max(neg_freqs) if len(neg_freqs) > 0 else 0
		max_freq = max(max_pos_freq, max_neg_freq)
		return 1 + log(max_freq)
		


	def compute_IDF(self, term, tf_info):
		"""
			idf(w, D) = log(card(D) / card(number of documents in which w occurs))
		"""

		nb_pos_review_for_term = tf_info[Utils.POS]["nb_review"] if Utils.POS in tf_info.keys() else 0
		nb_neg_review_for_term = tf_info[Utils.NEG]["nb_review"] if Utils.NEG in tf_info.keys() else 0
		nb_review_for_term = nb_pos_review_for_term + nb_neg_review_for_term

		nb_review = self.nb_pos_review + self.nb_neg_review
		return log(nb_review / nb_review_for_term)
