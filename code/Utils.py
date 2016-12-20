#!/usr/bin/python3

import os
import sys


######################
# Variables
######################

DB_ONE = 1
DB_TWO = 2
POS = 1
NEG = 0


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
