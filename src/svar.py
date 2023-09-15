
"""
Main implementation for the proposed ADMM-based SVAR estimation algorithm.

Copyright 2023, Jiahe Lin, Huitian Lei and George Michailidis
All Rights Reserved

Lin, Lei and Michailidis assert copyright ownership of this code base and its derivative
works. This copyright statement should not be removed or edited.

-----do not edit anything above this line---
"""

import numpy as np
from utils import graphUtil
from ._svarBase import _svarBase

class sVAR(_svarBase):
    
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
        return np.ones((p,p)).astype(float)
        
    def update_w(self, A, *args):
        return 1.0*(np.abs(A) < self.tau)

    def _postproc_A(self,A,Atilde,A_NZ):
        A = A * 1 * ((np.abs(A) > self.tau) & (np.abs(Atilde) > 1e-4))
        skeleton = 1 * (np.abs(A) != 0)
        for i in range(A.shape[0]):
            for j in range(i):
                
                if not (np.abs(A[i,j]) > self.tau*0.5 and np.abs(A[j,i]) > self.tau*0.5):
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
                        
        A = A * skeleton
        return A
