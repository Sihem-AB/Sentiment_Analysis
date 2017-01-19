#!/usr/bin/python3

import Preprocessing
import Utils
import TermFrequencyProcessing
import FileToReview


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

#############################################################################################
# 1st use case: When necessary json files are not created yet
#############################################################################################

print("\n1st scenario\n\n")

f2r = FileToReview.FileToReview(pos_path, neg_path, selected_DB)
pos_reviews, neg_reviews = f2r.buildReviewMatrix()

# get a new instance for preprocessing
# The new instance needs to know where positive and negative review directories are, also database no 
prep = Preprocessing.Preprocessing(pos_path, neg_path, selected_DB, pos_reviews, neg_reviews, is_bigrams)


# extract positive and negative vocabularies
prep.extract_vocabulary()
# print extracted vocabularies in dictionnary (json) format
vocabs = prep.get_v()

# get a new instance for processing
# The new instance needs to know where positive and negative review directories are, also database no 
# new json files will be created in the same directories
tfp = TermFrequencyProcessing.TermFrequencyProcessing(pos_path, neg_path, selected_DB)
tfp.compute_terms_frequency(vocabs)
print(tfp.get_overall_terms_frequency())

tfp.write_terms_frequency()


#############################################################################################
# 2nd use case: When necessary json files are already created
#############################################################################################

print("\n2nd scenario\n\n")

# get a new instance for processing
# The new instance needs to know where positive and negative review directories are, also database no 
tfp = TermFrequencyProcessing.TermFrequencyProcessing(pos_path, neg_path, selected_DB)

tfp.read_terms_frequency()
print(tfp.get_overall_terms_frequency())
