"""
Copyright 2023, Jiahe Lin, Huitian Lei and George Michailidis
All Rights Reserved

Lin, Lei and Michailidis assert copyright ownership of this code base and its derivative
works. This copyright statement should not be removed or edited.

-----do not edit anything above this line---
"""

import time
import warnings
import numpy as np

class pyAdmmUpdate():
    """
    Python implementation of the ADMM update
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
    
    def l2loss(self, X, Z, A, B):
        
        n = X.shape[0]
        loss = np.linalg.norm(X - np.matmul(X,A.transpose()) - np.matmul(Z,B.transpose()),'f')**2
        
        return loss/(2.0*n)
    
    def obj_primal(self, X, Z, w, A, Atilde, B, Btilde, xi, llambda, UA, UB, y, mu_A, mu_B):
        
        n, p = X.shape
        obj = self.l2loss(X,Z,A,B)
        obj += mu_A*np.sum(np.abs(Atilde)) + mu_B*np.sum(np.abs(Btilde))
        obj += self.rho/2*np.linalg.norm(A - Atilde,'f')**2 + self.rho*np.trace(np.matmul((A - Atilde).transpose(), UA))
        obj += self.rho/2*np.linalg.norm(B - Btilde,'f')**2 + self.rho*np.trace(np.matmul((B - Btilde).transpose(), UB))
        
        acyclic = 0
        for k in range(p):
            for i in range(p):
                for j in range(p):
                    if i != j:
                        temp = np.abs(Atilde[i,j])*w[i,j] + self.tau*(1-w[i,j]) + xi[i,j,k] - self.tau*llambda[i,k] - self.tau*1*(j!=k) + self.tau*llambda[j,k]
                        acyclic += (self.rho/2)*temp**2 + self.rho*temp*y[i,j,k]
        
        obj += acyclic
        return obj
        
    def set_M(self, p):
        
        M = np.zeros((p,p))
        for i in range(p):
            for j in range(p):
                if i == j:
                    M[i,j] = 1.0 + 1.0/(2*p)
                else:
                    M[i,j] = 1.0/(2*p)
                
        return M*(-1.0)/(2*self.tau*p)
                
    def update_A(self,X,V,A_NZ,Atilde,UA,A_LHSinv):
        
        n, p = X.shape
        XTV = np.matmul(X.transpose(),V)
        
        A = np.zeros((p,p))
        for i in range(p):
            nz = np.where(np.squeeze(A_NZ[i,:])!= 0)[0]
            RHS = XTV[nz,i]/n + self.rho*(Atilde[i,nz] - UA[i,nz])
            A[i,nz] = np.matmul(A_LHSinv[i,:len(nz),:len(nz)], RHS)
        return A

    def update_Atilde(self,A,UA,A_NZ,w,llambda,xi,y,mu_A):
        
        p = A.shape[0]
        Atilde = np.zeros(A.shape)
        ## collect pi
        pi = np.zeros((p,p,p))
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    pi[i,j,k] = xi[i,j,k] - self.tau*llambda[i,k] + self.tau*llambda[j,k] + y[i,j,k]
                    if j != k:
                        pi[i,j,k] -= self.tau
        
        ## update Atilde
        for i in range(p):
            for j in range(p):
                if A_NZ[i,j] == 0:
                    Atilde[i,j] = 0
                else:
                    sum_pi = np.sum(pi[i,j,:])
                    anchor = A[i,j]+UA[i,j]
                    if w[i,j] == 1:
                        Atilde[i,j] = np.sign(anchor)*np.maximum(0,(self.rho*np.abs(anchor)-self.rho*sum_pi-mu_A)/(1+p)/self.rho)
                    else:
                        Atilde[i,j] = np.sign(anchor)*np.maximum(0,np.abs(anchor)-mu_A/self.rho)
        return Atilde
    
    def update_UA(self, UA, A, Atilde):
        
        UA = UA + A - Atilde
        return UA
    
    def update_B(self,Z,W,Btilde,UB,B_LHSinv):
        
        n = Z.shape[0]
        RHS = np.matmul(W.transpose(),Z)/n + self.rho * (Btilde - UB)
        return np.matmul(RHS, B_LHSinv)
    
    def update_Btilde(self,B,UB,mu_B):
        
        Btilde = np.sign(B + UB) * np.maximum(0, np.abs(B+UB)-mu_B/self.rho)
        return Btilde
        
    def update_UB(self,UB,B,Btilde):
        
        UB = UB + B - Btilde
        return UB
    
    def update_lambda(self,A,Atilde,w,xi,y,M):
        
        p = A.shape[0]
        psi = np.zeros((p,p))
        
        for i in range(p):
            for k in range(p):
                sum1, sum2, sum3, sum4 = 0, 0, 0, 0
                sum_indicator1, sum_indicator2 = 0, 0
                for j in range(p):
                    if j!=i:
                        sum1 += np.abs(Atilde[j,i])*w[j,i] - np.abs(Atilde[i,j])*w[i,j]
                        sum2 += w[j,i] - w[i,j]
                        sum3 += xi[j,i,k] - xi[i,j,k]
                        sum4 += y[j,i,k] - y[i,j,k]
                        if i != k:
                            sum_indicator1 = sum_indicator1 + 1
                        if j != k:
                            sum_indicator2 = sum_indicator2 + 1
                psi[i,k] = sum1 - self.tau * sum2 + sum3 + sum4 - self.tau * (sum_indicator1 - sum_indicator2)
                
        llambda = np.matmul(M, psi)
        return llambda
    
    def update_xi(self,Atilde, w, llambda, y):
        
        p = Atilde.shape[0]
        xi = np.zeros((p,p,p))
        
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    temp = self.tau*llambda[i,k] - self.tau*llambda[j,k] - np.abs(Atilde[i,j]) * w[i,j] -y[i,j,k]-self.tau*(1-w[i,j])
                    if j!=k:
                        temp += self.tau
                    xi[i,j,k] = np.maximum(0, temp)
        return xi
    
    def update_y(self,y,A,Atilde,w,xi,llambda):
        
        p = A.shape[0]
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    y[i,j,k] += np.abs(Atilde[i,j])*w[i,j] + self.tau*(1-w[i,j]) + xi[i,j,k] - self.tau*llambda[i,k] + self.tau*llambda[j,k]
                    if j != k:
                        y[i,j,k] -= self.tau
        return y
    
    def admm_update(self, X, Z, w, mu_A, mu_B, A_NZ, A_LHSinv, B_LHSinv, A, B, Atilde, Btilde):
        
        start = time.time()
        
        n, p = X.shape
        d = Z.shape[1]
        
        M = self.set_M(p)
        UA, UB = A - Atilde, B - Btilde
        llambda, xi, y = np.zeros((p,p)), np.zeros((p,p,p)), np.zeros((p,p,p))
        
        l2loss_vals = []
        loss_cur = self.l2loss(X, Z, A, B)
        l2loss_vals.append(loss_cur)
        
        primal_vals = []
        primal_cur = self.obj_primal(X, Z, w, A, Atilde, B, Btilde, xi, llambda, UA, UB, y, mu_A, mu_B)
        primal_vals.append(primal_cur)
        
        CONVERGE = False
        for itr in range(self.max_admm_iter):
            
            loss_prev = loss_cur
            
            ## primal descent
            V = X - np.matmul(Z,B.transpose())
            A = self.update_A(X,V,A_NZ,Atilde,UA,A_LHSinv)
            Atilde = self.update_Atilde(A,UA,A_NZ,w,llambda,xi,y,mu_A)
            
            W = X - np.matmul(X,A.transpose())
            B = self.update_B(Z,W,Btilde,UB,B_LHSinv)
            Btilde = self.update_Btilde(B,UB,mu_B)
            
            llambda = self.update_lambda(A,Atilde,w,xi,y,M)
            xi = self.update_xi(Atilde, w, llambda, y)
            
            ## track primal objective
            primal_cur = self.obj_primal(X, Z, w, A, Atilde, B, Btilde, xi, llambda, UA, UB, y, mu_A, mu_B)
            primal_vals.append(primal_cur)
            
            ## dual ascent
            UA = self.update_UA(UA, A, Atilde)
            UB = self.update_UB(UB, B, Btilde)
            y = self.update_y(y,A,Atilde,w,xi,llambda)
            
            ## track l2loss, determine convergence
            loss_cur = self.l2loss(X,Z,A,B)
            
            if loss_cur > 1e4:
                print(f' >> admm_iter = {itr}, l2loss = {loss_cur:.2f} exceeds the pre-set explosion threshold at 1e4; force break')
                break
            
            if self.verbose > 0:
                if (itr +1) % self.verbose == 0:
                    print(f' >> admm_iter = {itr}, delta_loss = {loss_cur-loss_prev:.5f}, l2loss = {loss_cur:.2f}, primal = {primal_cur:.2f}')
            
            if np.abs(loss_cur - loss_prev) < self.admm_tol:
                CONVERGE = True
                break
                
            
        end = time.time()
        elapse = end - start
        
        if not CONVERGE:
            warnings.warn('*** admm fails to converge ***',category=UserWarning,stacklevel=2)
        else:
            if self.verbose:
                print(f'*** admm converges at iteration {itr+1}; time elapsed = {elapse:.2f} seconds ***')
            
        return {'A': A,
                'B':B,
                'Atilde': Atilde,
                'Btilde': Btilde,
                'l2loss': np.array(l2loss_vals),
                'primal': np.array(primal_vals)}
