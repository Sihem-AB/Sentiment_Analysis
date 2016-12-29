#!/usr/bin/python3


import os
import sys
from math import log

######################
# Variables
######################

DB_ONE = 1
DB_TWO = 2
POS = 1
NEG = 0

POS_RATING_DEFAULT = -1
NEG_RATING_DEFAULT = -1

######################
# Functions
######################

"""
	Check if path_dir is a directory (either local path or global path)
"""
def is_directory(path):
	if not(os.path.isdir(path)) and not( os.path.isdir(os.path.join(os.getcwd(), path)) ):
		return 0
	return 1



def is_file(path):
	if not(os.path.isfile(path)) and not( os.path.isfile(os.path.join(os.getcwd(), path)) ):
		return 0
	return 1


"""
	return the parent directory the file specified by "path"
	example: 
		path: "/home/ls.txt"
		output: "/home/"
"""
def get_parent_directory_for_file(path):
	if is_file(path):
		return "/".join(path.split("/")[:-1])
	return path



"""
	example:
		input: l = [("a", 1), ("b", 3)]
		output: {"a": 1, "b": 3}
"""
def make_dict_from_two_value_paired_list(l):
	d = dict()
	for item in l:
		d[item[0]] = item[1]
	return d



"""
	conditional mutual information and dealing with zero probabilities
    source: http://stats.stackexchange.com/questions/73502/conditional-mutual-information-and-how-to-deal-with-zero-probabilities

"""
def flexible_log(x):
	if x != 0:
		return log(x)
	return 0
		
	
