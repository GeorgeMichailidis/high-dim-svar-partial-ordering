"""
Base class for implementing the proposed ADMM-based SVAR estimation algorithm.

Copyright 2023, Jiahe Lin, Huitian Lei and George Michailidis
All Rights Reserved

Lin, Lei and Michailidis assert copyright ownership of this code base and its derivative
works. This copyright statement should not be removed or edited.

-----do not edit anything above this line---
"""


import time
import datetime
import warnings
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from ctypes import cdll
from sklearn.linear_model import Lasso
from .customENet import CustomENet

from .pyAdmmUpdate import pyAdmmUpdate
#from .cAdmmUpdate import cAdmmUpdate
#class _svarBase(cAdmmUpdate):
class _svarBase(pyAdmmUpdate):
    def __init__(
        self, 
        tau = 0.0001, 
        rho = 1, 
        max_admm_iter = 2000,
        admm_tol = 0.0001,
        verbose = 1,
        max_epoch = 30,
        tol = 0.01,
        threshold_A = 0.001,
        threshold_B = None,
        SILENCE = False
    ):
        """
        - tau (float), threshold for converting L0 norm to L1 truncated norm
        - rho (float): relaxation scaler for ADMM
        - max_iter (int): maximum number of iterations for the admm loop
        - admm_tol (float): convergence tolerance for ADMM
        - max_epoch (int): maximum number of iterations for the w <-> {A,B,\lambda} alternating update
        - tol (float): convergence tolerance for the DC step
        """
        self.SILENCE = SILENCE
        if SILENCE:
            verbose = 0
            
        super().__init__(tau, rho, admm_tol, max_admm_iter, verbose)
        self.tol = tol
        self.max_epoch = max_epoch
        self.threshold_A = threshold_A
        self.threshold_B = threshold_B
        
    def precalc_ALHSinv(self, X, A_NZ):
        
        n, p = X.shape
        A_LHSinv = np.zeros((p,p,p))
        for i in range(p):
            nz = np.where(np.squeeze(A_NZ[i,:])!= 0)[0]
            if len(nz) == 0:
                A_LHSinv[i,:,:] = np.zeros((p,p))
            else:
                coef_nz = np.linalg.inv((1.0/n)*np.matmul(X[:,nz].transpose(), X[:,nz]) + self.rho * np.identity(len(nz)))
                A_LHSinv[i,:len(nz),:len(nz)] = coef_nz
        return A_LHSinv

    def precalc_BLHSinv(self, Z):
        
        n, d = Z.shape
        B_LHSinv = np.linalg.inv((1.0/n)*np.matmul(Z.transpose(), Z) + self.rho * np.identity(d))
        return B_LHSinv
    
    def arrange_samples(self, xdata, q=2):
        
        X = xdata.copy()[q:,:]
        Z_unstack = [xdata.copy()[(q-i):(-i),:] for i in range(1,q+1)]
        Z = np.concatenate(Z_unstack, axis=1)
        
        assert X.shape[0] == Z.shape[0]
        assert Z.shape[1] == (xdata.shape[1] * q)
        
        return X, Z
    
    def proc_A_NZ(self, p, A_NZ = None):
        
        if A_NZ is None:
            A_NZ = 1 - np.identity(p,dtype=np.int32)
        else:
            A_NZ = A_NZ.astype(np.int32)
        assert np.array_equal(A_NZ.diagonal(), np.zeros((p,))), 'diagonal elements of A_NZ must be all zero'
        return A_NZ
    
    def calc_l2loss(self,X,Z,A,B):
        n = X.shape[0]
        loss = np.linalg.norm(X - np.matmul(X,A.transpose()) - np.matmul(Z,B.transpose()),'f')**2
        return loss/(2.0*n)
    
    def calc_costfunc(self,X,Z,A,B,mu_A,mu_B):
        """ (1/2n)|| X_n - X_nB^\top - X_{n-1}B^\top ||_F^2 + mu_A * |A|_1 + mu_B * |B|_1 """
        l2loss = self.calc_l2loss(X,Z,A,B)
        penalty = mu_A * np.sum(1*np.abs(A)) + mu_B * np.sum(1*np.abs(B))
        return l2loss + penalty
        
    def refit(self, xdata, A, q, mu_B_refit, rescale=True):
        
        X, Z = self.arrange_samples(xdata, q)
        if rescale:
            mu_B_refit *= np.sqrt(np.log(X.shape[1])/X.shape[0])
            if not self.SILENCE:
                print(f' * mu_B_refit has been scaled by sqrt(log(pq)/n); effecitve mu_B_refit = {mu_B_refit:.3f}')
        
        A_refit, B_refit = np.zeros((X.shape[1],X.shape[1])), np.zeros((X.shape[1], Z.shape[1]))
        model_refit = CustomENet(alpha = mu_B_refit,
                                 l1_ratio=1.0,
                                 standardize=False,
                                 fit_intercept=False,
                                 tol=1e-4,
                                 max_iter=10000,
                                 random_state=2023)
        
        for j in range(xdata.shape[1]):
            
            suppSet = np.where(np.abs(A[j,:])>0)[0]
            Znew = np.concatenate([X[:,suppSet],Z],axis=1)
            penalty = np.array([1] * Znew.shape[1])
            penalty[:len(suppSet)] = 0
            model_refit.fit(Znew, X[:,j], penalty)
            
            B_refit[j,:] = model_refit.w[len(suppSet):]
            A_refit[j,suppSet] = model_refit.w[:len(suppSet)]

        ## reshape
        B_refit = self._stack_B(B_refit)
        return A_refit, B_refit
    
    def fitSVAR(self, xdata, q=1, mu_A = 0.01, mu_B = 0.5, mu_B_refit = None, A_NZ = None):
        """
        main function for fitting an SVAR
        """
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Start fitting SVAR')
        ###############################
        ## 0: initial data prep
        ###############################
        X, Z = self.arrange_samples(xdata, q=q)
        n, p = X.shape
        d = int(p*q)
        ## convert mu_B to the scale of mu_B * sqrt(log (p)/n)
        mu_B_original = mu_B
        mu_B *= np.sqrt(np.log(d)/n)
        
        if not self.SILENCE:
            print(f'[Stage]: basic data manipulation')
            print(f' * input.shape={xdata.shape}; response(X).shape={X.shape}; lag(Z).shape={Z.shape}')
            print(f' * mu_B has been scaled by sqrt(log(pq)/n); effecitve mu_B = {mu_B:.3f}')
        
        ###############################
        ## 1: initialization and some pre-calc
        ###############################
        if not self.SILENCE:
            print(f'[Stage]: initialization with A = 0 and B = 0 ... ', end="")
        
        A = np.zeros((p,p))
        B = np.zeros((p,d))
        
        Atilde, Btilde = A.copy(), B.copy()
        w_prev = self.initialize_w(p)
        
        ## additional pre-calc
        A_NZ = self.proc_A_NZ(p, A_NZ)
        A_LHSinv = self.precalc_ALHSinv(X, A_NZ)
        B_LHSinv = self.precalc_BLHSinv(Z)
        
        if not self.SILENCE:
            print(f'Done!')
        
        ###############################
        ## 2: DC alternate update
        ###############################
        start = time.time()
        ## iterate
        if not self.SILENCE:
            print(f'[Stage]: start alternating update ... ')
        
        l2loss_tracker = [np.array(self.calc_l2loss(X,Z,A,B))] ## list of np.array
        cost_tracker = [self.calc_costfunc(X,Z,A,B,mu_A,mu_B)] ## list of float
        primal_tracker = [] ## list of np.array
        CONVERGE = False
        
        for epoch in range(self.max_epoch):
            
            admm_out = self.admm_update(X, Z, w_prev, mu_A, mu_B, A_NZ, A_LHSinv, B_LHSinv, A, B, Atilde, Btilde)
            
            A, B = admm_out['A'], admm_out['B']
            Atilde, Btilde = admm_out['Atilde'], admm_out['Btilde']
            
            ## tracking
            l2loss = admm_out['l2loss']
            l2loss_tracker.append(l2loss)
            
            ## calculate cost function
            cost = self.calc_costfunc(X,Z,A,B,mu_A,mu_B)
            cost_tracker.append(cost)
            
            primal = admm_out['primal']
            primal_tracker.append(primal)
            
            ## w update
            w_cur = self.update_w(A,w_prev,A_NZ)
            wdiff_pct = np.sum((w_cur - w_prev)**2)/(p*p)
            
            if not self.SILENCE:
                print(f' * epoch = {epoch+1}: l2loss = {l2loss[-1]:.4f}, cost = {cost:.4f}, wdiff = {wdiff_pct*100:.2f}%')
            
            ## determine convergence
            if wdiff_pct < self.tol:
                CONVERGE = True
                break
            
            w_prev = w_cur.copy()
        
        end = time.time()
        elapse = end - start
        
        if not self.SILENCE:
            if not CONVERGE:
                print(f'Algo status: FAIL TO CONVERGE within {self.max_epoch} epochs; TIME ELAPSED = {elapse:.3f} seconds')
                print(f'>> wdiff = {wdiff_pct*100:.3f}%; input mu_A = {mu_A:.2f}, mu_B = {mu_B_original:.2f}')
            else:
                print(f'Algo status: CONVERGE @ epoch = {epoch+1}; TIME ELAPSED = {elapse:.3f} seconds')
        
        ###############################
        ## 3: post-processing and refitting
        ###############################
        if not self.SILENCE:
            print(f'[Stage]: final touch up')
            
        self.check_constraint(A, A_NZ)
        A = self._postproc_A(A, Atilde, A_NZ)
        
        if mu_B_refit is not None:
            A, B = self.refit(xdata, A, q, mu_B_refit)
        else:
            B = self._postproc_B(B, Btilde)
        
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Done fitting SVAR')
        return {'A':A,
                'B':B,
                'Atilde': admm_out['Atilde'],
                'Btilde': admm_out['Btilde'],
                'Athreshold': self._threshold_A(A),
                'Bthreshold': self._threshold_B(B),
                'cost_tracker':cost_tracker,
                'l2loss_tracker': l2loss_tracker,
                'primal_tracker': primal_tracker}
    
    def predict(self, xdata, A, B):
        """
        get the predicted values similar to those in a regression
        Note that this is different from forecast in that this step essentially gets the in-sample fit
        """
        q = B.shape[-1]
        X, Z = self.arrange_samples(xdata, q=q)
        B_unstack = np.concatenate([B[:,:,i] for i in range(B.shape[-1])],axis=1)
        X_pred = np.matmul(X, A.transpose()) + np.matmul(Z, B_unstack.transpose())
        return X, X_pred
        
    def forecast(self, xdata, A, B, horizon):
        """conduct forecast based on a SVAR model"""
        B_reduced = np.zeros(B.shape)
        p,q = B.shape[0],B.shape[-1] ## number of lags
        I_mins_A_inv =  np.linalg.inv(np.identity(A.shape[1])-A)
        for i in range(q):
            B_reduced[:,:,i] = np.matmul(I_mins_A_inv, B[:,:,i])
        x_forecast_collect = [xdata[-(i+1)] for i in reversed(range(q))]
        for h in range(horizon):
            x_f = 0
            for i in range(q):
                x_f += np.dot(B_reduced[:,:,i],x_forecast_collect[-(i+1)])
            x_forecast_collect.append(x_f)
        return np.array(x_forecast_collect[-horizon:])
    
    ## check whether prior constraint has been respected
    def check_constraint(self, estimate, prior):
        for i in range(len(prior)):
            for j in range(i+1):
                if prior[i,j] == 0 and estimate[i,j] != 0:
                    warnings.warn(f'{(i,j)} fails to respect the prior constraint')
                if prior[j,i] == 0 and estimate[j,i] != 0:
                    warnings.warn(f'{(j,i)} fails to respect the prior constraint')

    ## post-procs
    def _stack_B(self,B):
        p, d = B.shape
        q = int(d/p)
        B_stacked = np.zeros((p,p,q))
        for i in range(q):
            B_stacked[:,:,i] = B[:,(p*i):(p*(1+i))]
        return B_stacked
    
    def _postproc_B(self,B,Btilde):
        B_new = B * 1 * (np.abs(Btilde) > 1e-4)
        B_stacked = self._stack_B(B_new)
        return B_stacked
    
    def _threshold_A(self,A):
        if self.threshold_A is not None:
            A_thresholded = 1*(np.abs(A) > self.threshold_A)*A
        else:
            A_thresholded = A.copy()
        return A_thresholded
        
    def _threshold_B(self,B):
        if self.threshold_B is not None:
            B_thresholded = 1*(np.abs(B) > self.threshold_B)*B
        else:
            B_thresholded = B.copy()
        return B_thresholded
