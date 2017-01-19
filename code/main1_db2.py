#!/usr/bin/python3

import Preprocessing
import Utils
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
V = prep.get_v()
print(V)

# write the vocabs into 2 json files in order to save vocabs in a structured form
prep.write_vocab()


#############################################################################################
# 2nd use case: When necessary json files are already created
#############################################################################################

print("\n2nd scenario\n\n")
# get a new instance for preprocessing
# The new instance needs to know where positive and negative review directories are, also database no 
prep = Preprocessing.Preprocessing(pos_path, neg_path, selected_DB, None, None, is_bigrams)

prep.read_vocab()
V = prep.get_v()
print(V)

