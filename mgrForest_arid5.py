#!/usr/bin/env python
# coding: utf-8
'''
code for arid6 dataset
you should change
    X = np.load('./X_17296_new.npy')
    y = np.load('./y_17296_new.npy')
to your own dataset path
than shell 'python mgrForest_arid5.py'

'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from itertools import product
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict as cvp
import random
from functools import reduce

class MultiGrainedScaner():
    def __init__(self, base_estimator, params_list, sliding_ratio = 0.25, k_fold = 2):
        if k_fold > 1: #use cv
            self.params_list = params_list
        else:#use oob
            self.params_list = [params.update({'oob_score':True}) or params for params in params_list]
        self.sliding_ratio = sliding_ratio
        self.k_fold = k_fold
        self.base_estimator = base_estimator
        klass = self.base_estimator.__class__
        self.estimators = [klass(**params) for params in self.params_list]

    #generate scaned samples, X is not None, X[0] is no more than 3d
    def _sample_slicer(self,X,y):
        data_shape = X[0].shape
        stride = 3
        window_shape = [max(int(data_size * self.sliding_ratio),1) for data_size in data_shape]
        scan_round_axis = [int((data_shape[i]-window_shape[i])/stride+1) for i in range(2)]
        scan_round_total = reduce(lambda acc,x: acc*x,scan_round_axis)
        if len(data_shape) == 1:
            newX = np.array([x[beg * window_shape[0]:(beg+1)*window_shape[0]]
                                for x in X
                                    for beg in range(scan_round_axis[0])])
        elif len(data_shape) == 2: #ravel 拉伸
            newX = np.array([x[beg0*stride:beg0*stride+window_shape[0],beg1*stride:beg1*stride+window_shape[1]].ravel()
                                for x in X
                                    for beg0 in range(scan_round_axis[0])
                                        for beg1 in range(scan_round_axis[1])])
        elif len(data_shape) == 3:
            newX = np.array([x[beg0 * stride:beg0 * stride + window_shape[0],beg1 * stride:beg1*stride + window_shape[1]].ravel()
                                for x in X
                                    for beg0 in range(scan_round_axis[0])
                                        for beg1 in range(scan_round_axis[1])])
        newy = y.repeat(scan_round_total)
        return newX,newy,scan_round_total

    #generate new sample vectors
    def scan_fit(self,X,y):
        self.n_classes = len(np.unique(y))
        newX,newy,scan_round_total = self._sample_slicer(X,y)
        sample_vector_list = []
        for estimator in self.estimators:
            estimator.fit(newX, newy)
            if self.k_fold > 1:# use cv
                predict_ = cvp(estimator, newX, newy, cv=self.k_fold, n_jobs = -1)
            else:#use oob
                predict_ = estimator.oob_decision_function_
                #fill default value if meet nan
                inds = np.where(np.isnan(predict_))
                predict_[inds] = 1./self.n_classes
            sample_vector = predict_.reshape((len(X),scan_round_total*self.n_classes))
            sample_vector_list.append(sample_vector)
        return np.hstack(sample_vector_list)

    def scan_predict(self,X):
        newX,newy,scan_round_total = self._sample_slicer(X,np.zeros(len(X)))
        sample_vector_list = []
        for estimator in self.estimators:
            predict_ = estimator.predict(newX)
            sample_vector = predict_.reshape((len(X),scan_round_total*self.n_classes))
            sample_vector_list.append(sample_vector)
        return np.hstack(sample_vector_list)
    


# cascade_params_list = [cascade_forest_params1,cascade_forest_params2]*2

# def calc_accuracy(pre,y):
#     return float(sum(pre==y))/len(y)
class ProbRandomForestClassifier(RandomForestClassifier):
    def predict(self, X):
        return RandomForestClassifier.predict_proba(self, X)
    


class RFLayer_RAND(object):
    def __init__(self, n_estimators, classifier=True , md=None, mss=10):
        self.n_estimators = n_estimators
        self.max_depth = md
        self.min_samples_split = mss
        self.classifier = classifier

    def fit(self, X_train, y_train, kfold=5, k=1, n_jobs=-1): # kfold = 5 yields 80/20 split, k will be the number of times we run validation
        if kfold > 1:
            kf = KFold(kfold, shuffle=True)
        else:
            raise ValueError('Need to pass kfold something greater than 1 so can do cross validation')

        models = []
        best_score = 0
        best_ind = 0
        count = 0

        # split training data into training and estimating sets via quasi kfold validation routine
        for tr_ind, est_ind in kf.split(X_train, y_train):
            # instantiate the layer of decision trees
            models.append(RandomForestClassifier(self.n_estimators, criterion='gini', max_depth=self.max_depth,
                                                 min_samples_split=self.min_samples_split,min_samples_leaf = 1,
                                                 max_features = 'sqrt',
                                                 n_jobs=n_jobs))
#             for tree in models[count].self.estimators_: # make half of the trees completely random Decision Trees
#             for tree in models[count].:
#                 if np.random.rand() <= .5:
#                     tree.splitter = 'random'


            # get the split of the training data
            X_tr, y_tr = X_train[tr_ind,:], y_train[tr_ind]
            # train the layer on this split
            models[count].fit(X_tr, y_tr)
            X_tr, y_tr = 0, 0

            # check accuracy on the estimation set
            X_est, y_est = X_train[est_ind,:], y_train[est_ind]
            y_pred = models[count].predict(X_est)
            acc_score = accuracy_score(y_pred, y_est)
            X_est, y_est = 0, 0 # memory
            y_pred = 0 # memory

            if acc_score > best_score: # with k > 1 we compare to see which is best layer trained
                best_score = acc_score
                best_ind = count
            count += 1
            if count >= k:
                break

        # save the best layer
        self.L = models[best_ind]
        self.n_classes = self.L.n_classes_
        self.val_score = best_score

    def predict(self, X_test):
        return self.L.predict(X_test)

    def push_thru_data(self, X):
        n_samples, dim_data = X.shape
        X_push = np.empty((n_samples, self.n_estimators*self.n_classes))
        # push the data X through this layer
        i = 0
        for tree in self.L.estimators_:
            if self.classifier:
                X_push[:,i*self.n_classes:(i+1)*self.n_classes] = tree.predict_proba(X)    
            i += 1
        X_a = np.concatenate((X_push,X[:,:self.n_estimators*self.n_classes]),axis = 1)
        return X_a




# In[ ]:
def main():
    X = np.load('./X_17296_new.npy')
    y = np.load('./y_17296_new.npy')
    # X = X.astype('float32') / 255.
    aa=15000
    X_train = X[:aa]
    y_train = y[:aa]
    X_test =  X[aa:]
    y_test =  y[aa:]
    X = 0
    y = 0
    scan_forest_params1 = RandomForestClassifier(n_estimators=10,min_samples_split=21,max_features='sqrt',n_jobs=-1).get_params()
    scan_forest_params2 = ExtraTreesClassifier(n_estimators = 10,min_samples_split=21, n_jobs=-1).get_params()
    scan_params_list = [scan_forest_params1,scan_forest_params2]
    Scaner1 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./2)
    Scaner2 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./4)

    # Scaner3 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./16)

    print('start training samples scanning.....')
    import time
    st = time.time()
    X_train_scan =np.hstack([scaner.scan_fit(X_train, y_train)
                                 for scaner in [Scaner1,Scaner2]])
    print(' training samples:',X_train_scan.shape)

    # train the next layers on multigrained scanning data
    print( 'RF Layer training:')


    # parameters for the building of the next layers
    n = 500# num trees in each layer
    min_gain = 0.01
    verbose = True
    max_layers = 6
    md = None
    mss = 21
    n_jobs = -1

    # dictionary where layers of decision trees will be stored
    Layers = {}

    prev_score = -1.0 # instantiate prev_score
    # build the layers
    for i in range(max_layers):
        print (X_train_scan.shape)
        RFL = RFLayer_RAND(n, md=md, mss=mss)
        RFL.fit(X_train_scan, y_train, 3, 1, n_jobs)
        Layers[i] = RFL

        # if verbose, print out the estimation accuracy for this layer
        if verbose:
            print ('Layer ' + str(i+1))
            print ('acc: ' + str(RFL.val_score))


        # check to see if we have improved enough going one more layer
        rel_gain = (RFL.val_score - prev_score)/float(abs(prev_score))
        if rel_gain < min_gain or RFL.val_score == 1.0 :
            print ('Converged! Stopping building layers')
            print
            break
        prev_score = RFL.val_score

        # if moving on to another level, push the data through
        X_train_scan = RFL.push_thru_data(X_train_scan)
        print ('Going to another layer')
        print
        
    et = time.time()
    print('training time:',et - st)


    print ('Loading in testing data')
    import time
    st = time.time()
    X_a_scan = np.hstack([scaner.scan_predict(X_test)
                                 for scaner in [Scaner1,Scaner2]])
    print ('Load over')
    # push test data thru FTDRF layers
    for i in range(len(Layers.keys())-1):
        X_a_scan = Layers[i].push_thru_data(X_a_scan)
    last = len(Layers.keys())-1
    y_pred = Layers[last].predict(X_a_scan)
    np.save('./pred.npy',y_pred)
    et = time.time()
    print('testing time:',et - st)

    print ('Statistics:')
    print ('The accuracy was:')
    print (accuracy_score(y_pred, y_test))
    print ('Params:')
    print ('num_tres in each layer = ' + str(n))
    print ('md =' + str(md))
    print ('mss = ' + str(mss))

if __name__ == "__main__":
    main()


