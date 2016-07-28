#!/usr/bin/python

import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from copy import deepcopy

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def modelfit(alg, train, test, predictors, target, \
             useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    """From Aarshay Jain's post Complete Guide to Parameter Tunning
    in XGBoost
    """
    # make a copy
    dtrain = deepcopy(train)
    dtest  = deepcopy(test)

    # Cross validation
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, \
                              label=dtrain[target].values, \
                              missing=float('nan'))
        xgtest = xgb.DMatrix(dtest[predictors].values, \
                             label=dtest[target].values, \
                             missing=float('nan'))
        cvresult = xgb.cv(xgb_param, xgtrain, \
                          num_boost_round=alg.get_params()['n_estimators'], \
                          nfold=cv_folds, \
                          metrics=['auc'], \
                          early_stopping_rounds=early_stopping_rounds, \
                          show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob    = alg.predict_proba(dtrain[predictors])[:,1]
    dtrain['Prediction'] = pd.Series(dtrain_predictions, index=dtrain.index)
    dtrain['predprob']   = pd.Series(dtrain_predprob, index=dtrain.index)

    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(
        dtrain[target].values, \
        dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(
        dtrain[target], \
        dtrain_predprob)

    #Predict on testing data:
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob    = alg.predict_proba(dtest[predictors])[:,1]
    dtest['Prediction'] = pd.Series(dtest_predictions, index=dtest.index)
    dtest['predprob']   = pd.Series(dtest_predprob, index=dtest.index)
    print "Accuracy : %.4g" % metrics.accuracy_score(
        dtest[target].values, \
        dtest_predictions)
    print 'AUC Score (Test): %f' % metrics.roc_auc_score(
        dtest[target], \
        dtest_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

    return dtrain, dtest

if __name__ == '__main__':
	# prepare data
	df = pd.read_csv('../data/train.csv')
	train, test = train_test_split(df, test_size=0.2)
	print train.shape, test.shape

	# prepare feature
	target = 'earn_over_4k_euros_per_year'
	index  = 'Id'
	predictors = [x for x in train.columns if x not in [target,index]]


	# parameter tuning
        # 0.8567
	#param = {'learning_rate'   : 0.1,
        #        'n_estimators'     : 1000,
        #        'max_depth'        : 2,
        #        'min_child_weight' : 1,
        #        'gamma'            : 0.4,
        #        'subsample'        : 0.9,
        #        'colsample_bytree' : 0.7,
        #        'objective'        : 'binary:logistic',
        #        'scale_pos_weight' : 1,
        #        'seed'             : 27}

	param = {'learning_rate'   : 0.1,
                'n_estimators'     : 120,
                'max_depth'        : 5,
                'min_child_weight' : 1,
                'gamma'            : 0.4,
                'subsample'        : 0.85,
                'colsample_bytree' : 0.5,
                'objective'        : 'binary:logistic',
                'scale_pos_weight' : 1,
                'seed'             : 27}

        param_tests = [
                #{'n_estimators':range(100,151,10)}
                #{'max_depth':range(3,10,2), 'min_child_weight':range(1,6,2)},
                {'max_depth':range(3,7,1), 'min_child_weight':range(1,3,1)},
                #{'gamma':[i/10.0 for i in range(0,7)]},
                #{'subsample':[i/100.0 for i in range(80,101,5)], \
                #        'colsample_bytree':[i/100.0 for i in range(50,101,10)]}
                #{'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
                #{'reg_alpha':[0, 0.01, 0.05, 0.1, 0.5]}

            ]

	for param_test in param_tests:
	    gsearch = GridSearchCV(
	        estimator = XGBClassifier(
	            learning_rate    = param['learning_rate'],
	            n_estimators     = param['n_estimators'],
	            max_depth        = param['max_depth'],
	            min_child_weight = param['min_child_weight'],
	            gamma            = param['gamma'],
	            subsample        = param['subsample'],
	            colsample_bytree = param['colsample_bytree'],
	            objective        = param['objective'],
	            scale_pos_weight = param['scale_pos_weight'],
	            seed             = param['seed']),
	        param_grid = param_test,
	        scoring='accuracy',
	        n_jobs=4,
	        iid=False,
	        cv=5)

	    gsearch.fit(train[predictors],train[target])
	    print gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

	    param.update(gsearch.best_params_)
