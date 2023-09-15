
"""
Alternative implementation for the proposed ADMM-based SVAR estimation algorithm

Copyright 2023, Jiahe Lin, Huitian Lei and George Michailidis
All Rights Reserved

Lin, Lei and Michailidis assert copyright ownership of this code base and its derivative
works. This copyright statement should not be removed or edited.

-----do not edit anything above this line---
"""

import numpy as np
from simulator import graphUtil
from ._svarBase import _svarBase

class sVARRaw(_svarBase):
    
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
        super().__init__(tau,rho,max_admm_iter,admm_tol,verbose,max_epoch,tol,threshold_A,threshold_B,SILENCE)
        
    def initialize_w(self,p):
        return np.ones((p,p)).astype(float)
        
    def update_w(self, A, *args):
        return 1.0*(np.abs(A) < self.tau)

    def _postproc_A(self,A,Atilde,*args):
        A_new = A * 1 * ((np.abs(A) > self.tau) & (np.abs(Atilde) > 1e-4))
        for i in range(A_new.shape[0]):
            for j in range(i):
                if (A_new[i,j] != 0) and (A_new[j,i] != 0):
                    if np.abs(A_new[i,j]) > np.abs(A_new[j,i]):
                        A_new[j,i] = 0
                    elif np.abs(A_new[i,j]) < np.abs(A_new[j,i]):
                        A_new[i,j] = 0
        return A_new

class sVARLT(_svarBase):
    
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
        super().__init__(tau,rho,max_admm_iter,admm_tol,verbose,max_epoch,tol,threshold_A,threshold_B,SILENCE)
        self.is_cyclic_ij = graphUtil().is_cyclic_ij
        
    def initialize_w(self,p):
        return np.identity(p).astype(float)
    
    def update_w(self,A,w_prev,A_NZ):
        
        skeleton = (1 - w_prev) * A_NZ
        for i in range(A.shape[0]):
            for j in range(i):
                    
                if not (np.abs(A[i,j])>self.tau*0.5 and np.abs(A[j,i])>self.tau*0.5):
                    continue
                    
                if np.abs(A[j,i]) > self.tau * 0.5:
                    if not self.is_cyclic_ij(skeleton,j,i) and A_NZ[j,i] == 1:
                        skeleton[j,i], skeleton[i,j] = 1, 0
                    elif A_NZ[i,j] == 1:
                        skeleton[i,j], skeleton[j,i] = 1, 0
                    else:
                        skeleton[i,j] = skeleton[j,i] = 0
                else:
                    if not self.is_cyclic_ij(skeleton,i,j) and A_NZ[i,j] == 1:
                        skeleton[i,j], skeleton[j,i] = 1, 0
                    elif A_NZ[j,i] == 1:
                        skeleton[j,i], skeleton[i,j] = 1, 0
                    else:
                        skeleton[i,j] = skeleton[j,i] = 0
        
        w = 1.0 - skeleton * A_NZ
        return w
        
    def _postproc_A(self,A,Atilde,*args):
        A_new = A * 1 * ((np.abs(A) > self.tau) & (np.abs(Atilde) > 1e-4))
        return A_new

