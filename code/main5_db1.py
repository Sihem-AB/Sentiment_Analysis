#!/usr/bin/python3

import Preprocessing
import Utils



"""
 # DB_ONE: Sentence Polarity Dataset
	This dataset is made up of 2 files: rt-polarity.neg and rt-polarity.pos	
	
 #DB_TWO: Large Movie Review Dataset
	This dataset is made up of 2 directories: pos/ and neg/. And each directory contains a number of review files
"""


pos_path = "../sampledata/dataset1/pos/rt-polarity.pos"
neg_path = "../sampledata/dataset1/neg/rt-polarity.neg"
selected_DB = Utils.DB_ONE
is_bigrams = True

#############################################################################################
# 1st use case: When necessary json files are not created yet
#############################################################################################


# get a new instance for preprocessing
# The new instance needs to know where positive and negative review directories are, also database no 
prep = Preprocessing.Preprocessing(pos_path, neg_path, selected_DB, is_bigrams)
# extract positive and negative vocabularies
prep.extract_vocabulary()
# print extracted vocabularies in dictionnary (json) format
V = prep.get_v()
#print(V)

# write the vocabs into 2 json files in order to save vocabs in a structured form
prep.write_vocab()


#############################################################################################
# 2nd use case: When necessary json files are already created
#############################################################################################

"""
# get a new instance for preprocessing
# The new instance needs to know where positive and negative review directories are, also database no 
prep = Preprocessing.Preprocessing(pos_path, neg_path, selected_DB, is_bigrams)

prep.read_vocab()
V = prep.get_v()
print(V)
"""
