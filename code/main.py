#!/usr/bin/python3

import Preprocessing
import Utils



"""
 # DB_ONE: Sentence Polarity Dataset
	This dataset is made up of 2 files: rt-polarity.neg and rt-polarity.pos	
	
 #DB_TWO: Large Movie Review Dataset
	This dataset is made up of 2 directories: pos/ and neg/. And each directory contains a number of review files
"""


#############################################################################################
# 1st use case
#############################################################################################

pos_path = "../sampledata/pos/"
neg_path = "../sampledata/neg/"

# get a new instance for preprocessing
# The new instance needs to know where positive and negative review directories are, also database no 
prep = Preprocessing.Preprocessing(pos_path, neg_path, Utils.DB_TWO)
# extract positive and negative vocabularies
prep.extract_vocab_DB_two()
# print extracted vocabularies in dictionnary (json) format
V = prep.get_v()
print(V)

# write the vocabs into 2 json files in order to save vocabs in a structured form
prep.write_vocab()



#############################################################################################
# 2nd use case
#############################################################################################

"""
pos_path = "/home/nejat/psud/information_extraction/code/pos/"
neg_path = "/home/nejat/psud/information_extraction/code/neg/"

# get a new instance for preprocessing
# The new instance needs to know where positive and negative review directories are, also database no 
prep = Preprocessing.Preprocessing(pos_path, neg_path, Utils.DB_TWO)

prep.read_vocab()
V = prep.get_v()
print(V)
"""
