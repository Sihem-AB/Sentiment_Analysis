#!/usr/bin/python3

import Utils
import TermFrequencyProcessing
from math import floor
from math import log
from copy import deepcopy


class FeatureSelection(object):
	def __init__(self, T, reviews_info, nb_neg_review, nb_pos_review, nb_word_in_neg_reviews, nb_word_in_pos_reviews):
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
# Bag of words model
###############################################################################

	"""
		Let’s assume that after pre-processing of the data, we have N distinct words
		 in the entire document. Also, suppose that the document contains R reviews. 
		Each review, is represented by a N -dimensional feature vector. The entries
		of these vectors correspond to the N words in the document and the feature values
		reflect the frequency of their corresponding words in the review. Alternatively, 
		we can construct a binary feature vector where 1 indicates the presence of the 
		corresponding word in the review and zero represents otherwise. 
		Another option is also to use Tf-IDF value instead of binary value or frequency.
		===> Se 7th section in the article

		Input:
			vocabs: vocabulary object. It supposed to be reduced vocabulary according to feature selection
            features_space: selected features
            vector_type = one of these values: "FREQ", "TF-IDF", "BINARY"
	"""
	def create_bag_of_words_model(self, vocabs, features_space, vector_type="FREQ"):
		model = {}
		model[Utils.POS] = []
		model[Utils.NEG] = []


		# features_space is not efficient to find the index of the term
		# features_space contains MI score for each term, and this is unnecessary for this task
		# replace MI score by index value so that we can acces the index of each term
		indexed_features_space = deepcopy(features_space)
		index = 0
		for term, score in indexed_features_space.items():
			indexed_features_space[term] = index
			index += 1
		nb_features = len(indexed_features_space)


		for sentiment_class in [Utils.NEG, Utils.POS]:
			reviews = vocabs[sentiment_class]["reviews"]

			for review in reviews:
				vec = self.create_review_vector(nb_features, indexed_features_space, review, vector_type)
				model[sentiment_class].append(vec)
			
		return model




	def create_review_vector(self, nb_features, indexed_features_space, review, vector_type):
		# create zeros vector
		vec = [0] * nb_features
	
		nb_word_in_review = review["nb_word"]
		sentences = review["sentences"]
		for sentence in sentences:
			for term, freq in sentence.items():
				index = indexed_features_space[term]
				
				if vector_type == "FREQ":
					vec[index] = freq
				elif vector_type == "TF-IDF":
					vec[index] = self.compute_tf_idf(term, freq, nb_word_in_review)
				else: # BINARY
					vec[index] = 1
		
		return vec




	def compute_tf_idf(self, term, freq, nb_word_in_review):
		# TODO In the article, the following formula is used: tf(w, d) = 1 + log(number of occurences of w in d)	
		tf = freq/nb_word_in_review

		# idf(w, D) = log(card(D) / card(number of documents in which w occurs))
		tf_info = self.T[term]
		nb_pos_review_for_term = tf_info[Utils.POS]["nb_review"] if Utils.POS in tf_info.keys() else 0
		nb_neg_review_for_term = tf_info[Utils.NEG]["nb_review"] if Utils.NEG in tf_info.keys() else 0
		nb_review_for_term = nb_pos_review_for_term + nb_neg_review_for_term

		nb_review = self.nb_pos_review + self.nb_neg_review
		idf = log(nb_review / nb_review_for_term)

		return tf*idf


###############################################################################
# Reduced Vocabs
###############################################################################

	"""		
		features_space will not probably contain all words in the corpus of review.
		This method removes words which are not	used in the features_space.

		Input: 
			vocabs: extracted vocabulary by an instance of the class Preprocessing
			features_space: feature space computed by feature utilty method like Mutual Information
	"""
	def reduce_vocabs(self, vocabs, features_space):
		reduced_vocabs = deepcopy(vocabs) # create a new dict object without linking to the old one

		# remove terms which are not representative according to the feature utility method
		for sentiment_class in [Utils.NEG, Utils.POS]:
			reviews = reduced_vocabs[sentiment_class]["reviews"]

			for review in reviews:
				self.reduce_review(review, features_space)

			# update reviews
			reduced_vocabs[sentiment_class]["reviews"] = reviews
		
		return reduced_vocabs



	"""
		features_space will not probably contain all words in the corpus of review.
		This method updates each sentence of the given review (i.e. list words with their frequency) by removing words which are not used in the features_space
	"""
	def reduce_review(self, review, features_space):
		sentences = review["sentences"]
	
		for sentence in sentences:
			terms_to_be_removed = []
			
			for term, freq in sentence.items():
				if term not in features_space:
					terms_to_be_removed.append(term)	
			
			for term in terms_to_be_removed:
				del sentence[term]
			


###############################################################################
# Mutual Information
###############################################################################

	"""
		Input: 
			k : 	pourcentage of words with largest value.
					Values are computed through feature utility method. 
					It should be between 0 and 1. ==> Float
			method: method name. ==> String.
					By now, there is only 1 feature utility method which is "MI" (Mutual Information)
			
		Returns k% terms with largest values


		Compute feature utility.
        According to the source below, there are 3 feature utility functions: mutual information, chi square test and frequency
        source: http://nlp.stanford.edu/IR-book/html/htmledition/feature-selection-1.html#fig:featselalg
        For now, there is only 1 available feature utility: Mutual Information (MI)
	"""
	def build_features_space(self, k=1, method = "MI"):
		if k > 1 or k <= 0:
			k = 1 # default value
		
		FU = None # Feature Utility
		if method == "MI":
			FU = self.compute_MI()
		
		nb_features = len(FU)
		# transform pourcentage into number
		top_k = floor(k * nb_features) # i.e floor(30.2) = 30
		print("MI: The number of terms with largest values according to the parameter 'k' ====>", top_k, "\n")
		
		# sort list by mutual information value in descending order
		# extract top k terms
		k_repr_terms = sorted(FU, key=lambda x: x[1], reverse=True)[:top_k]
		#print(k_repr_terms)

		# return a dict object whose key is a term and value is its MI score
		k_repr_terms = Utils.make_dict_from_two_value_paired_list(k_repr_terms) 
		# the dict object is not sorted by MI score # TODO is it important?
		return k_repr_terms

	

	def compute_MI(self):
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
			N_01 = self.nb_pos_review - N_11
			N_00 = self.nb_neg_review - N_10
			N__1 = N_01 + N_11
			N_1_ = N_10 + N_11
			N__0 = N_10 + N_00
			N_0_ = N_00 + N_01
			N = N_11 + N_10 + N_01 + N_00

            # TODO break a line for the formula below
			result = ((N_11/N) * Utils.flexible_log( (N*N_11)/(N_1_*N__1) )) + ((N_01/N) * Utils.flexible_log( (N*N_01)/(N_0_*N__1) )) + ((N_10/N) * Utils.flexible_log( (N*N_10)/(N_1_*N__0) )) + ((N_00/N) * Utils.flexible_log( (N*N_00)/(N_0_*N__0) )) 

			L.append((term, result))

		return L


