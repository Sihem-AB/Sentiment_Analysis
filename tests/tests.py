#!/usr/bin/python3

import sys
sys.path.insert(0, '../code')
import Preprocessing
import Utils

text = "Turgid dialogue, feeble characterization - Harvey Keitel a judge? He plays more like an off-duty hitman - and a tension-free plot conspire to make one of the unfunniest films of all time. You feel sorry for the cast as they try to extract comedy from a dire and lifeless script. Avoid!"


pos_path = "../sampledata/pos/"
neg_path = "../sampledata/neg/"

# get a new instance for preprocessing
# The new instance needs to know where positive and negative review directories are, also database no 
prep = Preprocessing.Preprocessing(pos_path, neg_path, Utils.DB_TWO)


#########################################################################
#########################################################################

print("########### Test 1 : divide into sentences############\n")

aSentence = prep.divide_into_sentences(text)
print('\n-----\n'.join(aSentence))


#########################################################################
#########################################################################

print("\n\n########### Test 2 : tokenize ############\n")

aSentence = prep.divide_into_sentences(text)
l = []
for sentence in aSentence:
	aWords = prep.tokenize(sentence)
	l.append(str(aWords))
print('\n-----\n'.join(l))


#########################################################################
#########################################################################

print("\n\n########### Test 3 : stemming ############\n")

aSentence = prep.divide_into_sentences(text)
l = []
for sentence in aSentence:
	aWords = prep.tokenize(sentence)
	aWords = prep.apply_stemming(aWords)
	l.append(str(aWords))
print('\n-----\n'.join(l))


#########################################################################
#########################################################################

print("\n\n########### Test 4 : remove punctuations ############\n")

aSentence = prep.divide_into_sentences(text)
l = []
for sentence in aSentence:
	aWords = prep.tokenize(sentence)
	aWords = prep.remove_punctuation(aWords)
	l.append(str(aWords))
print('\n-----\n'.join(l))


#########################################################################
#########################################################################

"""
print("\n\n########### Test 5 : negation handling ############\n")

aSentence = prep.divide_into_sentences(text)
l = []
for sentence in aSentence:
	aWords = prep.tokenize(sentence)
	aWords = prep.negation_handling(aWords)
	l.append(str(aWords))
print('\n-----\n'.join(l))
"""

