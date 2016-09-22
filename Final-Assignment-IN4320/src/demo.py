#!/usr/bin/python

import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV



if __name__ == '__main__':
	# prepare data
	df = pd.read_csv('../data/train.csv')

        # keep
        df_fill = pd.get_dummies(df, dummy_na=True)
        # mode
        #df_fill = df.fillna(method='ffill')
        #df_fill = df_fill.fillna(method='bfill')
        train = df_fill

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

        # 0.84
	#param = {'learning_rate'   : 0.1,
        #        'n_estimators'     : 120,
        #        'max_depth'        : 5,
        #        'min_child_weight' : 1,
        #        'gamma'            : 0.4,
        #        'subsample'        : 0.85,
        #        'colsample_bytree' : 0.5,
        #        'objective'        : 'binary:logistic',
        #        'scale_pos_weight' : 1,
        #        'seed'             : 27}

        param = {'learning_rate'   : 0.1,
                'n_estimators'     : 160,
                'max_depth'        : 6,
                'min_child_weight' : 1,
                'gamma'            : 0.3,
                'subsample'        : 0.8,
                'colsample_bytree' : 0.5,
                'objective'        : 'binary:logistic',
                'scale_pos_weight' : 1,
                'reg_alpha'        : 0.05,
                'seed'             : 27}



        param_tests = [
                {'n_estimators':range(100,201,20),
                #'max_depth':range(3,10,2), 'min_child_weight':range(1,6,2),
                #'max_depth':range(3,9,1),
                #'gamma':[i/10.0 for i in range(0,6)],
                #'subsample':[i/100.0 for i in range(50,101,5)], \
                #       'colsample_bytree':[i/100.0 for i in range(50,101,10)]
                #'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
                #'reg_alpha':[0, 0.01, 0.05, 0.1, 0.5]
                #'reg_alpha':[0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
                }

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
                    reg_alpha        = param['reg_alpha'],
	            seed             = param['seed']),
	        param_grid = param_test,
	        scoring='accuracy',
	        n_jobs=4,
	        iid=False,
	        cv=10)

	    gsearch.fit(train[predictors],train[target])
	    print gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

	    param.update(gsearch.best_params_)
