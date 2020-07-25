import os
import sys
import pandas as pd
import numpy as np
from scipy import sparse as ssp
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import datetime
from load_data import load_train_data, load_test_data
from util import calc_usample_len

args = sys.argv
if len(args) == 1:
        DIR = os.path.join('result', 'tmp')
elif len(args) == 2:
        DIR = os.path.join('result', args[1] + '.' + datetime.datetime.today().strftime('%Y%m%d'))
else:
        print ('too many arguments')
        exit   
        
if not os.path.exists(DIR):
        os.mkdir(DIR)

SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'

logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(os.path.join(DIR, 'train.py.log'), 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)


def gini(y, pred):
        fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
        g = 2 * auc(fpr, tpr) -1
        return g

def gini_xgb(pred, y):
        y = y.get_label()
        return '-gini', -gini(y, pred)


def undersampling(df, desired_apriori=0.1):
        # Get the indices per target value
        idx_0 = df[df.target == 0].index
        idx_1 = df[df.target == 1].index

        # Get original number of records per target value
        nb_0 = len(df.loc[idx_0])
        nb_1 = len(df.loc[idx_1])
        
        logger.info('before: (0, 1)=({}, {})'.format(nb_0, nb_1))

        # Calculate the undersampling rate and resulting number of records with target=0
        undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
        undersampled_nb_0 = int(undersampling_rate*nb_0)
        logger.info('Rate to undersample records with target=0: {}'.format(undersampling_rate))
        logger.info('after: (0, 1)=({}, {})'.format(undersampled_nb_0, nb_1))

        # Randomly select records with target=0 to get at the desired a priori
        undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

        # Construct list with remaining indices
        idx_list = list(undersampled_idx) + list(idx_1)

        # Return undersample data frame
        df = df.loc[idx_list].reset_index(drop=True)
        return df

def check_null(df):
        vars_with_missing = []
        for f in df.columns:
                    missings = df[df[f] == -1][f].count()
                    if missings > 0:
                            vars_with_missing.append(f)
                            missings_perc = missings/df.shape[0]
                            logger.debug('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
        logger.debug('In total, there are {} variables with missing values'.format(len(vars_with_missing)))

def fill_null(df):
        # Dropping the variables with too many missing values
        vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
        df = df.drop(vars_to_drop, axis=1)

        # Imputing with the mean or mode
        mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
        mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
        df['ps_reg_03'] = mean_imp.fit_transform(df[['ps_reg_03']]).ravel()
        df['ps_car_12'] = mean_imp.fit_transform(df[['ps_car_12']]).ravel()
        df['ps_car_14'] = mean_imp.fit_transform(df[['ps_car_14']]).ravel()
        df['ps_car_11'] = mode_imp.fit_transform(df[['ps_car_11']]).ravel()
        return df


def preprocessing(df_train, df_test):
        #check_null(df)
        #logger.debug('start null insert')
        #df_full = fill_null(df_train)
        
        logger.debug('start undersampling')
        df_train = undersampling(df_train, desired_apriori=0.1)

        x_train = df_train.drop('target', axis=1)
        y_train = df_train['target'].values
        use_cols = x_train.columns.values    
        x_test = df_test[use_cols].sort_values('id')

        logger.debug('start num_features')    
        num_features = [c for c in use_cols if ('cat' not in c and 'calc' not in c)]
        x_train['missing'] = (x_train==-1).sum(axis=1).astype(float)
        x_test['missing'] = (x_test==-1).sum(axis=1).astype(float)
        num_features.append('missing')

        logger.debug('start cat_features')
        cat_features = [c for c in use_cols if ('cat' in c and 'count' not in c)]
        for c in cat_features:
                le = LabelEncoder()
                le.fit(x_train[c])
                x_train[c] = le.transform(x_train[c])
                x_test[c] = le.transform(x_test[c])
                                
        enc = OneHotEncoder(sparse=False)
        enc.fit(x_train[cat_features])
        X_train_cat = enc.transform(x_train[cat_features])
        X_test_cat = enc.transform(x_test[cat_features])
                
        logger.debug('start ind_features')
        ind_features = [c for c in use_cols if 'ind' in c]
        count=0
        for c in ind_features:
                if count==0:
                        x_train['new_ind'] = x_train[c].astype(str)+'_'
                        x_test['new_ind'] = x_test[c].astype(str)+'_'
                        count+=1
                else:
                        x_train['new_ind'] += x_train[c].astype(str)+'_'
                        x_test['new_ind'] += x_test[c].astype(str)+'_'
                        
        logger.debug('start cat_count_features')    
        cat_count_features = []
        for c in cat_features+['new_ind']:
                d_train = x_train[c].value_counts().to_dict()
                d_test = x_test[c].value_counts().to_dict()
                x_train['%s_count'%c] = x_train[c].apply(lambda x:d_train.get(x,0))              
                x_test['%s_count'%c] = x_test[c].apply(lambda x:d_test.get(x,0))
                cat_count_features.append('%s_count'%c)
        
        x_train_list = [x_train[num_features+cat_count_features].values,X_train_cat,]
        x_test_list = [x_test[num_features+cat_count_features].values,X_test_cat,]

        X_train = np.hstack(x_train_list)
        X_test = np.hstack(x_test_list)
        return X_train, y_train, X_test
        

if __name__=='__main__':
        logger.debug('start load data')
        df_train = load_train_data()
        df_test = load_test_data()
        X_train, y_train, X_test = preprocessing(df_train, df_test)
                               
        '''
        all_params = {'max_depth': [3, 5, 7],
                      'learning_rate': [0.1],
                      'min_child_weight': [3, 5, 10],
                      'n_estimators': [10000],
                      'colsample_bytree': [0.8, 0.9],
                      'colsample_bylevel': [0.8, 0.9],
                      'reg_alpha': [0, 0.1],
                      'max_delta_step': [0.1],
                      'seed': [0]}
        '''
        
        all_params = {'max_depth': [7],
                      'learning_rate': [0.1],
                      'min_child_weight': [5],
                      'n_estimators': [10000],
                      'colsample_bytree': [0.8],
                      'colsample_bylevel': [0.8],
                      'reg_alpha': [0],
                      'max_delta_step': [0.1],
                      'seed': [0]}

        n_bag = 1
        desired_apriori = 0.1
        logger.debug('all params: {}'.format(all_params))
        logger.debug('n_bag: {}'.format(n_bag))
        logger.debug('desired_apriori: {}'.format(desired_apriori))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        
        min_score = 100
        min_params = None
        grid_cnt = 0
        logger.debug('start grid search')
        for params in tqdm(list(ParameterGrid(all_params))):
                logger.debug('{}/{}'.format(grid_cnt, len(list(ParameterGrid(all_params)))))
                logger.debug('params: {}'.format(params))
                                
                list_gini_score = []
                list_logloss_score = []
                list_best_iterations = []
                for train_idx, valid_idx in cv.split(X_train, y_train):
                        trn_y = y_train[train_idx]
                        idx_0 = train_idx[trn_y == 0]
                        idx_1 = train_idx[trn_y == 1]

                        
                        sc_logloss = 0
                        sc_gini = 0
                        sample_len = calc_usample_len(len(idx_1), len(idx_0), desired_apriori)
                        for i in range(n_bag):
                                # Randomly select records with target=0 to get at the desired a priori
                                new_idx_0 = shuffle(idx_0, random_state=37, n_samples=sample_len)
                                idx_list = list(new_idx_0) + list(idx_1)
                             
                                # Construct list with remaining indices
                                _X_train = X_train[idx_list]                 
                                _y_train = y_train[idx_list]
                                
                                _X_valid = X_train[valid_idx]
                                _y_valid = y_train[valid_idx]
                
                                clf = xgb.sklearn.XGBClassifier(**params)
                                clf.fit(_X_train, _y_train, eval_set=[(_X_valid, _y_valid)], early_stopping_rounds=100, eval_metric=gini_xgb)
                                
                                pred = clf.predict_proba(_X_valid, ntree_limit=clf.best_ntree_limit)[:, 1]
                                sc_logloss += log_loss(_y_valid, pred) / n_bag
                                sc_gini += - gini(_y_valid, pred) / n_bag
                        
                        list_logloss_score.append(sc_logloss)
                        list_gini_score.append(sc_gini)
                        list_best_iterations.append(clf.best_iteration)            
                        break
                
                params['n_estimators'] = int(np.mean(list_best_iterations))
                sc_logloss = np.mean(list_logloss_score)                
                sc_gini = np.mean(list_gini_score)
                if min_score > sc_gini:
                        min_score = sc_gini
                        min_params = params
                logger.debug('logloss: {}, gini: {}'.format(sc_logloss, sc_gini))
                grid_cnt += 1

        logger.debug('minimum params: {}'.format(min_params))
        logger.debug('minimum gini: {}'.format(min_score))
        logger.debug('end grid search')
                            
        clf = xgb.sklearn.XGBClassifier(**min_params)
        clf.fit(X_train, y_train)

        #df = load_test_data()

        #logger.debug('load test data start')
        #x_test = df[use_cols].sort_values('id')
        
        '''
        for col in use_cols:
                if col not in df.columns:
                        logger.info('{} is not in test data'.format(col))
                        df[col] = np.zeros(df.shape[0])
        '''
        #logger.debug('load test data end')

        logger.debug('predict test data start')
        pred_test = clf.predict_proba(X_test)[:, 1]
        logger.debug('predict test data end')

        df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
        df_submit['target'] = pred_test

        df_submit.to_csv(os.path.join(DIR, 'submission.csv'), index=False)
        logger.info('end')
                                                               
