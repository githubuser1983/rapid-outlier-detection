#!/usr/bin/python

import sys,csv,math
sys.path.insert(1,'/usr/local/lib/python2.7/dist-packages/')

import sklearn
print sklearn.__version__
import time, pandas as pd
print pd.__version__
import numpy as np
from rapidoutlierdetection import RapidOutlierDetection
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors.lof import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

class Benchmark(object):
    """This class provides a way to benchmark different outlier detection algorithms on different datasets.
    """
    def __init__(self,alg_name, algorithm, dataset):
        self.alg_name = alg_name
        self.algorithm = algorithm
        self.dataset = dataset

    def start(self):
        X = dataset[[x for x in dataset.columns[0:-1]]]
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        ground_truth = dataset["cl"]*(-2)+1
        start = time.time()
        if self.alg_name == 'lof':
            y_pred = algorithm.fit_predict(X)
        else:
            algorithm.fit(X)
            y_pred = algorithm.predict(X)
        tp = ((y_pred==-1) & (ground_truth==-1)).sum()
        fp = ((y_pred==-1) & (ground_truth==1)).sum()
        fn = ((y_pred==1) & (ground_truth==-1)).sum()
        print (self.alg_name, tp,fp,fn,tp+fp,tp+fn)
        end = time.time()
        precision = tp*1.0/(1.0*tp+fp)
        recall = tp * 1.0/(1.0*tp+fn)
        return end-start, precision, recall

def read_datasets():
    datasets = { 
                 'covtype' : './data/covtype_original.csv',
                 'gaussian1e3':'./data/gaussian1e3_original.csv'
                 'kdd1999' : './data/kdd1999_original.csv'
                }
    dataframes = []
    for name,filename in datasets.items():
        dataframes.append( (name, pd.read_csv(filename,sep=";",header=0)) )
    return dataframes

if __name__ == "__main__":

    num_experiments = 10
    times = dict([])
    precisions  = dict([])
    recalls = dict([])

    datasets = read_datasets()
    for i in range(0,num_experiments):
        np.random.seed(i)
        rng = np.random.RandomState(i)
        for dataname, dataset in datasets:
            n_samples = dataset.shape[0]
            n_features = dataset.shape[1]
            contamination = dataset["cl"].sum()*1.0/(n_samples*1.0)
            print dataname, n_samples,n_features,contamination
            algorithms = {
                "Rapid Outlier Detection": RapidOutlierDetection(contamination=contamination,scaled=False,random_state=rng), # the benchmark scales ..
                "One-Class SVM": svm.OneClassSVM(nu=0.95 *contamination + 0.05,kernel="rbf", gamma=0.1),
                "Isolation Forest": IsolationForest(max_samples=n_samples,contamination=contamination,random_state=rng),
                 "lof": LocalOutlierFactor(n_neighbors=35,contamination=contamination)
                  }
            for algorithmname, algorithm in algorithms.items():
                benchmark = Benchmark(algorithmname,algorithm,dataset)
                difftime,precision, recall = benchmark.start()
                print dataname, algorithmname,difftime,precision,recall
                times[(algorithmname,dataname,i)] = difftime
                precisions[(algorithmname,dataname,i)] = precision
                recalls[(algorithmname,dataname,i)] = recall
                  
print times
print precisions
print recalls
