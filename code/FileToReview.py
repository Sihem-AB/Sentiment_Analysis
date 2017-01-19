#!/usr/bin/python3

import Utils
import numpy as np
import os
import sys
import re


class FileToReview(object):
	def __init__(self, pos_path, neg_path, selected_DB):
		self.pos_path = pos_path 
		self.neg_path = neg_path
		self.selected_DB = selected_DB


	def buildReviewMatrix(self):
		pos_reviews = None
		neg_reviews = None

		if self.selected_DB == Utils.DB_ONE:
			pos_reviews = self.buildReviewMatrixDB1(self.pos_path)
			neg_reviews = self.buildReviewMatrixDB1(self.neg_path)
		elif self.selected_DB == Utils.DB_TWO:
			pos_reviews = self.buildReviewMatrixDB2(self.pos_path)
			neg_reviews = self.buildReviewMatrixDB2(self.neg_path)

		return pos_reviews, neg_reviews 


	"""
		Reviews in DB1 dont contain any rating information. 
			That is why we use default rating value, which is Utils.RATING_DEFAULT

		This type of matrix reviews contains both reviews and their ratings. 
		In addition, it is easy to use Cross Validation methods with it.

		example of output:

		array([['not bad :) ', '-1'],
	       	['Loved it', '-1'],
    	  	[' I can be pretty picky but loved it!', '-1'],
       		['Enjoy enjoy the show!', '-1']], 
      		dtype='<U1')
	"""
	def buildReviewMatrixDB1(self, path):
		mReviews = []


		if not( Utils.is_file(path) ):
			print("error: path is not an existing file")
			# TODO raise an error
			return 0

		with open("./" + path, "r") as f:
			for sReview in f:
				row = [sReview, Utils.RATING_DEFAULT]
				mReviews.append(row)

		return np.array(mReviews) 


	"""
		Reviews in DB2 contain rating information.

		This type of matrix reviews contains both reviews and their ratings. 
		In addition, it is easy to use Cross Validation methods with it.
		
		example of output:

		
		array([['not bad :) ', '6'],
	       	['Loved it', '9'],
    	  	[' I can be pretty picky but loved it!', '8'],
       		['Enjoy enjoy the show!', '7']], 
      		dtype='<U1')
	"""
	def buildReviewMatrixDB2(self, path):
		mReviews = []

		if not( Utils.is_directory(path) ):
			print("error: path is not a directory")
			# TODO raise an error
			return 0

		# get only .txt files and not .json files
		files = [f for f in os.listdir(path) if re.match(r'.*\.txt', f)]

		for filename in files:
			with open (path+"/"+filename, "r", encoding="utf8") as f:
				sReview = f.read()
				rating = self.extract_rating(filename)
				row = [sReview, rating]
				mReviews.append(row)

		return np.array(mReviews)



	def extract_rating(self, filename):
		# a rating value could be 10 so 2 digit
		part = filename.split("_")[1]
		return part.split(".")[0] # rating
