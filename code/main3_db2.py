#!/usr/bin/python3

import Utils
import Preprocessing
import TermFrequencyProcessing
import FeatureSelection


"""
 # DB_ONE: Sentence Polarity Dataset
	This dataset is made up of 2 files: rt-polarity.neg and rt-polarity.pos	
	
 #DB_TWO: Large Movie Review Dataset
	This dataset is made up of 2 directories: pos/ and neg/. And each directory contains a number of review files
"""

pos_path = "../sampledata/dataset2/pos/"
neg_path = "../sampledata/dataset2/neg/"
selected_DB = Utils.DB_TWO
is_bigrams = False
method = "MI"

#############################################################################################
# 1st use case: when necessary json files are not created yet
#############################################################################################


prep = Preprocessing.Preprocessing(pos_path, neg_path, selected_DB, is_bigrams)
# extract positive and negative vocabularies
prep.extract_vocabulary()
# print extracted vocabularies in dictionnary (json) format
vocabs = prep.get_v()


# get a new instance
# The new instance needs to know where positive and negative review directories are, also database no 
tfp = TermFrequencyProcessing.TermFrequencyProcessing(pos_path, neg_path, selected_DB)

tfp.compute_terms_frequency(vocabs)
#print(tfp.get_overall_terms_frequency())
#print(tfp.get_reviews_info())
T = tfp.get_overall_terms_frequency()
reviews_info = tfp.get_reviews_info()

nb_neg_review = tfp.get_nb_neg_review()
nb_pos_review = tfp.get_nb_pos_review()
nb_word_in_neg_reviews = tfp.get_nb_word_in_neg_reviews()
nb_word_in_pos_reviews = tfp.get_nb_word_in_pos_reviews()



fs = FeatureSelection.FeatureSelection(T, reviews_info, nb_neg_review, nb_pos_review, nb_word_in_neg_reviews, nb_word_in_pos_reviews)
k = 0.2 # top k% terms
print(fs.build_features_space(k, method))


#############################################################################################
# 2nd use case: when necessary json files are already created
#############################################################################################

"""
# get a new instance
# The new instance needs to know where positive and negative review directories are, also database no 

tfp = TermFrequencyProcessing.TermFrequencyProcessing(pos_path, neg_path, selected_DB)
tfp.read_terms_frequency()
T = tfp.get_overall_terms_frequency()
tfp.read_reviews_info()
reviews_info = tfp.get_reviews_info()

nb_neg_review = tfp.get_nb_neg_review()
nb_pos_review = tfp.get_nb_pos_review()
nb_word_in_neg_reviews = tfp.get_nb_word_in_neg_reviews()
nb_word_in_pos_reviews = tfp.get_nb_word_in_pos_reviews()



fs = FeatureSelection.FeatureSelection(T, reviews_info, nb_neg_review, nb_pos_review, nb_word_in_neg_reviews, nb_word_in_pos_reviews)
k = 0.2 # top k% terms
print(fs.build_features_space(k, method))
"""
