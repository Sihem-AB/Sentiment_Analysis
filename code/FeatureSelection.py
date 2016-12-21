#!/usr/bin/python3

import Utils
import TermFrequencyProcessing
from math import floor


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
		Input: 
			k : pourcentage of words with largest MI value. 
				It should be between 0 and 1. ==> Float
			
		Returns k% terms with largest values


		Compute feature utility for DB_Two.
        According to the source below, there are 3 feature utility function: mutual information, chi square test and frequency
        source: http://nlp.stanford.edu/IR-book/html/htmledition/feature-selection-1.html#fig:featselalg
        For now, there is only 1 available feature utility: Mutual Information (MI)
	"""
	def compute_feature_utility_DB_two(self, k=0.2):
		L = self.compute_MI(self.nb_neg_review, self.nb_pos_review)

		if k > 1 or k <= 0:
			k = 0.2 # default value

		# transform pourcentage into number
		top_k = floor(k * (self.nb_word_in_neg_reviews + self.nb_word_in_pos_reviews)) # i.e floor(30.2) = 30
		print("The number of terms with largest values according to the 'k' parameter ====>", top_k, "\n")
		
		# sort list by mutual information value in descending order
		# extract top k terms
		return sorted(L, key=lambda x: x[1], reverse=True)[:top_k]




	def compute_MI(self, nb_neg_review, nb_pos_review):
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
