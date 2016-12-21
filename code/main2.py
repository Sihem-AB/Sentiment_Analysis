#!/usr/bin/python3

import Preprocessing
import Utils
import TermFrequencyProcessing


"""
 # DB_ONE: Sentence Polarity Dataset
	This dataset is made up of 2 files: rt-polarity.neg and rt-polarity.pos	
	
 #DB_TWO: Large Movie Review Dataset
	This dataset is made up of 2 directories: pos/ and neg/. And each directory contains a number of review files
"""


#############################################################################################
# 1st use case: When necessary json files are not created yet
#############################################################################################

pos_path = "../sampledata/pos/"
neg_path = "../sampledata/neg/"


# get a new instance for processing
# The new instance needs to know where positive and negative review directories are, also database no 
tfp = TermFrequencyProcessing.TermFrequencyProcessing(pos_path, neg_path, Utils.DB_TWO)
tfp.compute_terms_frequency()
#print(tfp.get_overall_terms_frequency())
#print(tfp.get_reviews_info())

tfp.write_terms_frequency()
tfp.write_reviews_info()


#############################################################################################
# 2nd use case: When necessary json files are already created
#############################################################################################

"""
pos_path = "../sampledata/pos/"
neg_path = "../sampledata/neg/"

# get a new instance for processing
# The new instance needs to know where positive and negative review directories are, also database no 
tfp = TermFrequencyProcessing.TermFrequencyProcessing(pos_path, neg_path, Utils.DB_TWO)

tfp.read_terms_frequency()
print(tfp.get_overall_terms_frequency())
tfp.read_reviews_info()
print(tfp.get_reviews_info())
"""
