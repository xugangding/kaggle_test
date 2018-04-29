

from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from sklearn import naive_bayes
from sklearn import preprocessing
from scipy import sparse
from itertools import combinations

import sets
import numpy as np
import pandas as pd
import sys

def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        if 5 in indicies and 7 in indicies:
            print ("feature Xd")
        elif 2 in indicies and 3 in indicies:
            print ("feature Xd")
        else:
            new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T
def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap
