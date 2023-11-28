"""
Copyright 2023, Jiahe Lin and George Michailidis
All Rights Reserved

Lin and Michailidis assert copyright ownership of this code base and its derivative
works. This copyright statement should not be removed or edited.

-----do not edit anything above this line---
"""

import time
import warnings
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from ctypes import cdll

class cAdmmUpdate():
    """
    Python-Cpp implementation of performing ADMM block update.
    To create shared library
    g++ -c -o cAdmmUpdate.o cAdmmUpdate.cpp
    g++ -shared -o cAdmmUpdate.so cAdmmUpdate.o
    """
    def __init__(self, tau = 0.0001, rho = 1, admm_tol = 0.0001, max_admm_iter = 2000, verbose=1):
        """
        - tau (float), threshold for converting L0 norm to L1 truncated norm
        - rho (flaot): relaxation scaler for ADMM
        - max_admm_iter (int): maximum number of iterations for the admm loop
        """
        self.tau = float(tau)
        self.rho = float(rho)
        self.admm_tol = float(admm_tol)
        self.max_admm_iter = int(max_admm_iter)
        self.verbose = int(verbose)
        
        ## handle function inputs
        lib = ctypes.cdll.LoadLibrary('./src/cAdmmUpdate.so')
        func = lib.cAdmmUpdate
        func.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## loss
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## primal objective
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## A
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## B
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## Atilde
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## Btilde
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## w
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## X
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## Z
                         ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),    ## A_NZ
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## A_LHSinv
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ## B_LHSinv
                         ctypes.c_double, ## mu_A
                         ctypes.c_double, ## mu_B
                         ctypes.c_double, ## rho
                         ctypes.c_double, ## tau
                         ctypes.c_double, ## admm_tol
                         ctypes.c_int,    ## n
                         ctypes.c_int,    ## p
                         ctypes.c_int,    ## d
                         ctypes.c_int,    ## maxIter
                         ctypes.c_int,    ## verbose
                        ]
        self.admm_func = func
    
    def admm_update(self, X, Z, w, mu_A, mu_B, A_NZ, A_LHSinv, B_LHSinv, A, B, Atilde, Btilde):
        
        n, p = X.shape
        d = Z.shape[1]
        
        ## pre-allocate memory to track l2loss and primal objective
        l2loss_vals = (-1.0)*np.ones((self.max_admm_iter+1,))
        primal_vals = (-1.0)*np.ones((self.max_admm_iter+1,))
        
        CONVERGE = self.admm_func(l2loss_vals,primal_vals,A,B,Atilde,Btilde,w,X,Z,A_NZ,A_LHSinv,B_LHSinv,mu_A,mu_B,self.rho,self.tau,self.admm_tol,n,p,d,self.max_admm_iter,self.verbose)
        
        if not CONVERGE:
            warnings.warn('*** admm fails to converge ***',category=UserWarning,stacklevel=2)
        
        ## only keep the loss up to the valid iteration
        l2loss_vals = list(filter(lambda x: x >= 0, l2loss_vals))
        primal_vals = list(filter(lambda x: x >= 0, primal_vals))
        
        return {'A': A,
                'B':B,
                'Atilde': Atilde,
                'Btilde': Btilde,
                'l2loss': np.array(l2loss_vals),
                'primal': np.array(primal_vals)}
