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

pos_path = "../sampledata/dataset1/pos/rt-polarity.pos"
neg_path = "../sampledata/dataset1/neg/rt-polarity.neg"
selected_DB = Utils.DB_ONE
is_bigrams = False
method = "MI"
vector_type = "TF-IDF"
# vector_type = "FREQ"

#############################################################################################
# 1st use case: when necessary json files are not created yet
#############################################################################################

print("\n1st scenario\n\n")

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
T = tfp.get_overall_terms_frequency()

nb_neg_review = prep.get_nb_neg_review()
nb_pos_review = prep.get_nb_pos_review()



fs = FeatureSelection.FeatureSelection(T,  nb_neg_review, nb_pos_review)
k = 0.2 # top k% terms
features_space = fs.build_features_space(k, method)
reduced_vocabs = fs.reduce_vocabs(vocabs, features_space)

bag_of_words_model = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type)
print(bag_of_words_model)


#############################################################################################
# 2nd use case: when necessary json files are already created
#############################################################################################


print("\n2nd scenario\n\n")

# get a new instance
# The new instance needs to know where positive and negative review directories are, also database no 

prep = Preprocessing.Preprocessing(pos_path, neg_path, selected_DB, is_bigrams)
# extract positive and negative vocabularies
prep.extract_vocabulary()
vocabs = prep.get_v()

tfp = TermFrequencyProcessing.TermFrequencyProcessing(pos_path, neg_path, selected_DB)
tfp.read_terms_frequency()
T = tfp.get_overall_terms_frequency()

nb_neg_review = prep.get_nb_neg_review()
nb_pos_review = prep.get_nb_pos_review()


fs = FeatureSelection.FeatureSelection(T, nb_neg_review, nb_pos_review)
k = 0.2 # top k% terms
features_space = fs.build_features_space(k, method)
reduced_vocabs = fs.reduce_vocabs(vocabs, features_space)

bag_of_words_model = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type)
print(bag_of_words_model)

