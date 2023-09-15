"""
Evaluator

Copyright 2023, Jiahe Lin, Huitian Lei and George Michailidis
All Rights Reserved

Lin, Lei and Michailidis assert copyright ownership of this code base and its derivative
works. This copyright statement should not be removed or edited.

-----do not edit anything above this line---
"""


import numpy as np

class Evaluator():
    def __init__(self):
        self.metrics = ['sensitivity',
                        'specificity', 
                        'fpr', 
                        'fdr', 
                        'precision',
                        'recall',
                        'accuracy', 
                        'mcc', 
                        'f1score',  
                        'err_fnorm_rel']
    
    def tp(self, truth, est):
        return np.sum(1 * (np.abs(est) > 0) & (np.abs(truth)>0))
        
    def tn(self, truth, est):
        return np.sum(1 * (np.abs(est) == 0) & (np.abs(truth) == 0))
    
    def fp(self, truth, est):
        return np.sum(1 * (np.abs(est) > 0) & (np.abs(truth) == 0))
    
    def fn(self, truth, est):
        return np.sum(1 * (np.abs(est) == 0) & (np.abs(truth) > 0))
    
    def sensitivity(self, truth, est):
        return self.tp(truth, est)/(self.tp(truth,est) + self.fn(truth,est))
    
    def specificity(self, truth, est):
        return self.tn(truth, est)/(self.tn(truth,est) + self.fp(truth,est))
    
    def fpr(self, truth, est):
        return 1 - self.specificity(truth, est)
    
    def fdr(self, truth, est):
        return self.fp(truth,est)/(self.fp(truth,est) + self.tp(truth, est))
    
    def precision(self, truth, est):
        return 1 - self.fdr(truth, est)
    
    def recall(self, truth, est):
        return self.sensitivity(truth, est)
    
    def accuracy(self, truth, est):
        return (self.tp(truth, est) + self.tn(truth, est))/est.size
        
    def mcc(self, truth, est):
        tp, fp = self.tp(truth, est), self.fp(truth,est)
        tn, fn = self.tn(truth, est), self.fn(truth,est)
        denominator = np.sqrt(tp+fp) * np.sqrt(tp+fn) * np.sqrt(tn+fp) * np.sqrt(tn+fn)
        return (tp*tn - fp*fn)/denominator

    def f1score(self, truth, est):
    
        precision = self.precision(truth,est)
        recall = self.recall(truth, est)
        return 2*precision*recall/(precision+recall)
        
    def err_fnorm_rel(self, truth, est):
        return np.linalg.norm(est-truth, 'f')/np.linalg.norm(truth,'f')
    
    def gather_metric(self, truth, est):
        
        self.metric_container = {}
        for metric in self.metrics:
            fn = getattr(self, metric)
            self.metric_container[metric] = fn(truth, est)
        self.metric_container['point_on_ROC'] = (self.metric_container['fpr'], self.metric_container['sensitivity'])
        
    def report(self, truth, est):
        self.gather_metric(truth, est)
        report = {'tpr':round(self.metric_container['sensitivity'],3),
                  'tnr':round(self.metric_container['specificity'],3)}
        return report
    
