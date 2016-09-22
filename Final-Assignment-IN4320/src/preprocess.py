#!/usr/bin/python

import numpy as np
from sklearn import preprocessing

def category2vector(categories):
    """Convert category data to binary vectors
    Parameters
    ----------
    categories : array of categories (n_samples,)

    Returns
    -------
    vectors : array of binary vectors representing the categories (n_samples, n_categories)

    Examples
    --------
    from preprocess import category2vector
    cs = ['a', 'b', 'b', 'c']
    vs = category2vector(cs)
    """

    # encode categories
    le = preprocessing.LabelEncoder()
    le.fit(categories)
    codes = le.transform(categories)

    # convert to binary vectors


