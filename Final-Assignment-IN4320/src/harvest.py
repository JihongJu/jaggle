#!/usr/bin/python

import os
import argparse
import pandas as pd
import numpy as np



def arg_parser():
    """Parse the text file names for train set and test.
    Returns
    -------
    fntrain : file name of train set (path included)
    fntest  : file name of test set  (path included)
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', help='text file name of train set')
    parser.add_argument('--test', help='text file name of test set')
    args = parser.parse_args()

    fntrain = args.train
    fntest  = args.test

    return fntrain, fntest

def import_data(fndata):
    """Import data from text files.
    Parameters
    ----------
    fndata : raw data file including path
    """
    with open(fndata, 'rb') as f:
        # split lines
        lsdata = [line.split(',') for line in f.read().splitlines()]
        # map to float
        lsdata = [map(float, row) for row in lsdata]

    # use numpy array
    arrdata = np.array(lsdata)

    return arrdata


def formulate_data(arrdata):
    """Formulate data to pandas format
    Parameters
    ---------
    arrdata : array of raw data

    Returns
    -------
    dfdata : structured data as DataFrame

    Attributes
    ----------
    Id : key
    age : continuous
    work class : categorical
    education : categorical
    number of years of education : continuous
    marital status : categorical
    occupation : categorical
    relationship : categorical
    race : categorical
    sex : categorical
    income from investment sources : continuous
    losses from investment sources : continuous
    working hours per week : continuous
    native country : categorical
    earn over 4k euros per year: binary (target)
    """

    N, K = arrdata.shape

    dfdata = pd.DataFrame({
        'Id'                            : np.arange(1,N+1).astype('int32'),
        'age'                           : arrdata[:,0].astype('float64'),
        'work_class'                    : pd.Categorical(arrdata[:,1]),
        'education'                     : pd.Categorical(arrdata[:,2]),
        'number_of_years_of_education'  : arrdata[:,3].astype('float64'),
        'marital_status'                : pd.Categorical(arrdata[:,4]),
        'occupation'                    : pd.Categorical(arrdata[:,5]),
        'relationship'                  : pd.Categorical(arrdata[:,6]),
        'race'                          : pd.Categorical(arrdata[:,7]),
        'sex'                           : pd.Categorical(arrdata[:,8]),
        'income_from_investment_sources': arrdata[:,9].astype('float64'),
        'losses_from_investment_sources': arrdata[:,10].astype('float64'),
        'working_hours_per_week'        : arrdata[:,11].astype('float64'),
        'native_country'                : pd.Categorical(arrdata[:,12])})

    # if earn_over_4k_euros_per_year is available (training set)
    if K == 14:
        dfdata['earn_over_4k_euros_per_year'] = \
                pd.Series(arrdata[:,13].astype('int32'), index=dfdata.index)
    else:
        dfdata['earn_over_4k_euros_per_year'] = \
                pd.Series((np.ones(N)*(-1)).astype('int32'), index=dfdata.index)

    return dfdata


if __name__ == '__main__':
    # parse arguments
    fntrain, fntest = arg_parser()
    dirname = os.path.dirname(fntrain)
    print 'Reading training set from {}'.format(fntrain)
    print 'Reading test set from {}'.format(fntest)

    # import data from file
    arrtrain = import_data(fntrain)
    arrtest  = import_data(fntest)


    # formulate the data
    dftrain = formulate_data(arrtrain)
    dftest  = formulate_data(arrtest)

    # save to csv files
    csvtrain = os.path.join(dirname, 'train.csv')
    csvtest  = os.path.join(dirname, 'test.csv')

    dftrain.to_csv(csvtrain, index=False)
    dftest.to_csv(csvtest, index=False)



