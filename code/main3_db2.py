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

pos_path = "../sampledata/dataset2/train/pos/"
neg_path = "../sampledata/dataset2/train/neg/"
selected_DB = Utils.DB_TWO
is_bigrams = False
method = "MI"

#############################################################################################
# 1st use case: when necessary json files are not created yet
#############################################################################################


print("\n1st scenario\n\n")

prep = Preprocessing.Preprocessing(pos_path, neg_path, selected_DB, is_bigrams)
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
#print(tfp.get_overall_terms_frequency())
T = tfp.get_overall_terms_frequency()



fs = FeatureSelection.FeatureSelection(T, nb_neg_review, nb_pos_review)
k = 0.2 # top k% terms
print(fs.build_features_space(k, method))


#############################################################################################
# 2nd use case: when necessary json files are already created
#############################################################################################

print("\n2nd scenario\n\n")

# get a new instance
# The new instance needs to know where positive and negative review directories are, also database no 

prep = Preprocessing.Preprocessing(pos_path, neg_path, selected_DB, is_bigrams)
prep.read_vocab()

nb_neg_review = prep.get_nb_neg_review()
nb_pos_review = prep.get_nb_pos_review()


tfp = TermFrequencyProcessing.TermFrequencyProcessing(pos_path, neg_path, selected_DB)
tfp.read_terms_frequency()
T = tfp.get_overall_terms_frequency()



fs = FeatureSelection.FeatureSelection(T, nb_neg_review, nb_pos_review)
k = 0.2 # top k% terms
print(fs.build_features_space(k, method))

