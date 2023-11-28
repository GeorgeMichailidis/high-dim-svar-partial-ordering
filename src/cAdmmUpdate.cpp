//************************************************************************
//
// Cpp implementation for the proposed ADMM block update
// Copyright 2023, Jiahe Lin and George Michailidis
// All Rights Reserved

// Lin and Michailidis assert copyright ownership of this code base and its derivative
// works. This copyright statement should not be removed or edited.
//
// -----do not edit anything above this line---
//
//************************************************************************

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <cassert>

using namespace std;

#define MIN(a,b) ((a) < (b) ?  (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ABS(a) ((a) < 0 ? -(a) : (a))
#define SIGN(a) ((a)< 0?(-1):((a) == 0 ? 0:1))

extern "C" {

void MatMat(double *out, double *Mat1, double *Mat2, int m1, int n1, int m2, int n2)
{
    if (n1 != m2)
        std::cout<<"Dimensions don't match."<< std::endl;
    assert (n1==m2);
    
    for (int i=0; i<m1; i++)
    {
        for (int j=0; j<n2; j++)
        {
            out[i*n2+j] = 0;
            for (int k=0; k<n1;k++)
            {
                out[i*n2+j] += Mat1[i*n1+k]*Mat2[k*n2+j]; // sum_over_k A[i,k] * B[k,j]
            }
        }
    }
}

void transpose(double *AT, double *A, int m, int n)
{
    // A is m*n
    for (int pos=0; pos<m*n; pos++)
    {
        int i = pos/n;
        int j = pos%n;
        AT[j*m+i] = A[pos];
    }
}

void calc_and_set_V(double *V, double *X, double *Z, double *B, int n, int p, int d)
{
    double *BT = new double[d*p];
    transpose(BT, B, p, d);
    double *ZBT = new double[n*p];
    MatMat(ZBT,Z,BT,n,d,d,p);
    for (int pos = 0; pos < n*p; pos++)
        V[pos] = X[pos] - ZBT[pos];
    
    delete [] BT;
    delete [] ZBT;
}

void calc_and_set_W(double *W, double *X, double *A, int n, int p)
{
    double *AT = new double[p*p];
    transpose(AT, A, p, p);
    double *XAT = new double[n*p];
    MatMat(XAT,X,AT,n,p,p,p);
    
    for (int pos = 0; pos < n*p; pos++)
        W[pos] = X[pos] - XAT[pos];
    
    delete [] AT;
    delete [] XAT;
}

void calc_and_set_M(double *M, int p, double tau)
{
    int i,j;
    for (i=0;i<p;i++)
    {
        for (j=0;j<p;j++)
        {
            if (j==i)
                M[i+j*p] = (-1.0)*(1.0 + 1.0/(2.0*p))/(2.0*tau*p);
            else
                M[i+j*p] = (-1.0)/(2.0*p)/(2.0*tau*p);
        }
    }
}

void calc_and_set_XTV(double *XTV, double *X, double *V, int n, int p)
{
    double *XT = new double[n*p];
    transpose(XT, X, n, p);
    MatMat(XTV, XT, V, p, n, n, p);
    
    delete [] XT;
}

void calc_and_set_ZTW(double *ZTW, double *Z, double *W, int n, int p, int d)
{
    double *ZT = new double[d*n];
    transpose(ZT, Z, n, d);
    MatMat(ZTW, ZT, W, d, n, n, p);
    
    delete [] ZT;
}

double l2loss(double *X, double *Z, double *A, double *B, int n, int p, int d)
{
    double *AT = new double[p*p];
    transpose(AT, A, p, p);
    double *XAT = new double[n*p];
    MatMat(XAT,X,AT,n,p,p,p);
    
    double *BT = new double[d*p];
    transpose(BT, B, p, d);
    double *ZBT = new double[n*p];
    MatMat(ZBT,Z,BT,n,d,d,p);
    
    double loss = 0.0;
    for (int pos=0; pos<n*p; pos++)
        loss += (X[pos] - XAT[pos] - ZBT[pos]) * (X[pos] - XAT[pos] - ZBT[pos]);
    
    delete [] AT;
    delete [] XAT;
    delete [] BT;
    delete [] ZBT;
    
    return loss/(2.0*n);
}

double inner_product(double *Mat1, double *Mat2, int m, int n)
{
    double *Mat1T = new double[n*m];
    transpose(Mat1T, Mat1, m, n);
    
    double *Mat1TMat2 = new double[n*n];
    MatMat(Mat1TMat2,Mat1T,Mat2,n,m,m,n);
    
    double trace = 0.0;
    for (int i=0; i < n; i ++)
        trace += Mat1TMat2[i*n+i];
    
    delete [] Mat1T;
    delete [] Mat1TMat2;
    
    return trace;
}

double delta_fnorm_sq(double *Mat1, double *Mat2, int m, int n)
{
    double delta_sum = 0;
    for (int pos = 0; pos < m*n; pos ++)
        delta_sum += (Mat1[pos] - Mat2[pos]) * (Mat1[pos] - Mat2[pos]);
    return delta_sum;
}

double obj_primal(double *X, double *Z, double *w, double *A, double *Atilde, double *B, double *Btilde, double *xi, double *lambda, double *UA, double *UB, double *y, double mu_A, double mu_B, double tau, double rho, int n, int p, int d)
{
    
    int pos;
    double obj = 0.0;
    obj += l2loss(X, Z, A, B, n, p, d);
    for (pos=0; pos <p*p; pos++)
        obj += mu_A * ABS(Atilde[pos]);
    for (pos=0; pos <p*d; pos++)
        obj += mu_B * ABS(Btilde[pos]);

    double *diff_AAtilde = new double[p*p];
    for (pos=0; pos < p*p; pos++)
        diff_AAtilde[pos] = A[pos] - Atilde[pos];
    obj += rho/(2.0) * delta_fnorm_sq(A, Atilde, p, p) + rho*inner_product(diff_AAtilde, UA, p, p);

    double *diff_BBtilde = new double[p*d];
    for (pos=0; pos < p*d; pos++)
        diff_BBtilde[pos] = B[pos] - Btilde[pos];
    obj += rho/(2.0) * delta_fnorm_sq(B, Btilde, p, d) + rho*inner_product(diff_BBtilde, UB, p, d);
    
    double acyclic = 0.0;
    double temp;
    for (int k=0; k < p; k++)
        for (int i=0; i<p; i++)
            for (int j=0; j<p; j++)
                if (i!=j)
                {
                    temp = ABS(Atilde[i*p+j])*w[i*p+j] + tau*(1-w[i*p+j]) + xi[i*p*p+j*p+k] - tau*lambda[i*p+k] + tau*lambda[j*p+k];
                    if (j!=k)
                        temp -= tau;
                    acyclic += rho/(2.0)*temp*temp + rho*temp*y[i*p*p+j*p+k];
                }

    obj += acyclic;
    return obj;
}

void update_A(double *A, double *Atilde, double *UA, int *A_NZ, double *XTV, double *A_LHSinv, double rho, int n, int p)
{
    double *row_RHS = new double[p];
    double *row_LHSinv = new double[p*p];
    double *row_Sol = new double[p];
    
    int i, j, nz, pos;
    for (i=0; i<p; i++)
    {
        nz = 0;
        for (j=0; j<p; j++)
        {
            row_RHS[j] = 0;
            if (A_NZ[i*p+j]==1)
            {
                row_RHS[nz] = XTV[j*p+i]*(1.0/n) + rho*(Atilde[i*p+j] - UA[i*p+j]);
                nz ++;
            }
        }
        
        for (pos=0; pos<p*p; pos++)
            row_LHSinv[pos] = A_LHSinv[i*p*p+pos];
        
        MatMat(row_Sol,row_LHSinv,row_RHS, p, p, p, 1);
        
        nz = 0;
        for (j=0; j<p; j++)
        {
            if (A_NZ[i*p+j]==0)
                A[i*p+j] = 0;
            else
            {
                A[i*p+j] = row_Sol[nz];
                nz ++;
            }
        }
    }
    
    delete [] row_RHS;
    delete [] row_LHSinv;
    delete [] row_Sol;
}

void update_Atilde(double *Atilde, double *A, double *UA, int *A_NZ, double *w, double *lambda, double *xi, double *y, double tau, double rho, double mu_A, int p)
{
    double *pi = new double[p*p*p];
    int i, j, k;
    
    for (i=0; i<p; i++)
    {
        for (j=0; j<p; j++)
        {
            for (k=0; k<p; k++)
            {
                pi[i*p*p+j*p+k] = 0;
                pi[i*p*p+j*p+k] = xi[i*p*p+j*p+k] - tau*lambda[i*p+k] + tau*lambda[j*p+k] + y[i*p*p+j*p+k];
                if (j!=k)
                {
                    pi[i*p*p+j*p+k] = pi[i*p*p+j*p+k] - tau;
                }
            }
        }
    }
    
    double sumk_pi, anchor;
    for (i=0; i<p; i++)
    {
        for (j=0; j<p; j++)
        {
            if (A_NZ[i*p+j]==0)
                Atilde[i*p+j] = 0;
            else
            {
                sumk_pi = 0;
                for (k=0; k<p; k++)
                    sumk_pi += pi[i*p*p+j*p+k];
                
                anchor = A[i*p+j]+UA[i*p+j];
                if (w[i*p+j]==1)
                    Atilde[i*p+j] = SIGN(anchor) * MAX(0,(rho*ABS(anchor)-rho*sumk_pi-mu_A)/(1+p)/rho);
                else
                    Atilde[i*p+j] = SIGN(anchor) * MAX(0,ABS(anchor) - mu_A/rho);
            }
        }
    }
    delete [] pi;
}

void update_UA(double *UA, double *A, double *Atilde, int p)
{
    for (int pos=0; pos < p*p; pos ++)
    {
        UA[pos] = UA[pos] + A[pos] - Atilde[pos];
    }
}

void update_B(double *B, double *Btilde, double *UB, double *ZTW, double *B_LHSinv, double rho, int n, int p, int d)
{
    double *RHS = new double[p*d];
    double *WTZ = new double[p*d];
    transpose(WTZ, ZTW, d, p);
    
    for (int pos=0; pos < p*d; pos ++)
        RHS[pos] = WTZ[pos]*(1.0)/n + rho*(Btilde[pos] - UB[pos]);
    
    MatMat(B, RHS, B_LHSinv, p, d, d, d);
    delete [] RHS;
    delete [] WTZ;
}

void update_Btilde(double *Btilde, double *B, double *UB, double rho, double mu_B, int p, int d)
{
    for (int i=0; i<p; i++)
    {
        for (int j=0; j<d; j++)
        {
            Btilde[i*d+j] = SIGN(B[i*d+j]+UB[i*d+j]) * MAX(0,ABS(B[i*d+j]+UB[i*d+j]) - mu_B/rho);
        }
    }
}

void update_UB(double *UB, double *B, double *Btilde, int p, int d)
{
    for (int pos=0; pos < p*d; pos ++)
        UB[pos] = UB[pos] + B[pos] - Btilde[pos];
}

void update_lambda(double *lambda, double *A, double *Atilde, double *w, double *xi, double *y, double *M, double tau, int p)
{
    double *Psi = new double[p*p];
    double sum1, sum2, sum3, sum4, sum_ind1, sum_ind2;
    int ijk, jik, ij, ji;
    
    for (int i=0; i<p; i++)
    {
        for (int k=0; k<p; k++)
        {
            sum1 = sum2 = sum3 = sum4 = 0.0;
            sum_ind1 = sum_ind2 = 0.0;
            for (int j=0; j<p; j++)
            {
                ijk = i*p*p+j*p+k; jik = j*p*p+i*p+k; ij = i*p+j; ji = j*p+i;
                if (j!=i)
                {
                    sum1 += ABS(Atilde[ji])*w[ji] - ABS(Atilde[ij])*w[ij];
                    sum2 += w[ji] - w[ij];
                    sum3 += xi[jik] - xi[ijk];
                    sum4 += y[jik] - y[ijk];
                    if (i!=k)
                        sum_ind1 += 1.0;
                    if (j!=k)
                        sum_ind2 += 1.0;
                }
            }
            Psi[i*p+k] = sum1 - tau*sum2 + sum3 + sum4 - tau*(sum_ind1 - sum_ind2);
        }
    }
    
    MatMat(lambda,M,Psi,p,p,p,p);
    delete [] Psi;
}

void update_xi(double *xi, double *Atilde, double *w, double *lambda, double *y, double tau, int p)
{
    int ijk, ij, ik, jk;
    double temp;
    for (int i=0; i<p; i++)
    {
        for (int j=0; j<p; j++)
        {
            for (int k=0; k<p; k++)
            {
                ijk = i*p*p+j*p+k; ij = i*p+j; ik = i*p+k; jk = j*p+k;
                temp = tau*lambda[ik] - tau*lambda[jk] - ABS(Atilde[ij]) * w[ij] - y[ijk] - tau*(1-w[ij]);
                if (j!=k)
                    temp += tau;
                xi[ijk] = MAX(0,temp);
            }
        }
    }
}

void update_y(double *y, double *A, double *Atilde, double *w, double *xi, double *lambda, double tau, int p)
{
    int ijk, ij, ik, jk;
    for (int i=0; i<p; i++)
    {
        for (int j=0; j<p; j++)
        {
            for (int k=0; k<p; k++)
            {
                ijk = i*p*p+j*p+k; ij = i*p+j; ik = i*p+k; jk = j*p+k;
                y[ijk] += ABS(Atilde[ij])*w[ij] + tau*(1-w[ij]) + xi[ijk] - tau*lambda[ik] + tau*lambda[jk];
                if (j!=k)
                    y[ijk] -= tau;
            }
        }
    }
}

bool cAdmmUpdate(double *l2loss_vals, double *primal_vals, double *A, double *B, double *Atilde, double *Btilde, double *w, double *X, double *Z, int *A_NZ, double *A_LHSinv, double *B_LHSinv, double mu_A, double mu_B, double rho, double tau, double tol, int n, int p, int d, int maxIter, int verbose)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    int pos;
    
    double *lambda = new double[p*p];
    for (pos=0; pos<p*p; pos++)
        lambda[pos] =0;
    
    double *xi = new double[p*p*p];
    for (pos=0; pos<p*p*p;pos++)
        xi[pos] = 0;
        
    double *UA = new double[p*p];
    for (pos=0; pos<p*p; pos++)
        UA[pos] = A[pos] - Atilde[pos];
        
    double *UB = new double[p*d];
    for (pos=0; pos<p*d; pos++)
        UB[pos] = B[pos] - Btilde[pos];
    
    double *y = new double[p*p*p];
    for (pos=0; pos<p*p*p;pos++)
        y[pos] = 0;
    
    double *M = new double[p*p];
    calc_and_set_M(M, p, tau);
    
    double *V = new double[n*p];
    double *W = new double[n*p];
    double *XTV = new double[p*p];
    double *ZTW = new double[d*p];
    
    bool converge = false;
    int iter;
    double delta_AAtilde, delta_BBtilde, loss_cur, loss_prev, delta_loss, primal_cur;
    
    loss_cur = l2loss(X, Z, A, B, n, p, d);
    l2loss_vals[0] = loss_cur;
    
    primal_cur = obj_primal(X, Z, w, A, Atilde, B, Btilde, xi, lambda, UA, UB, y, mu_A, mu_B, tau, rho, n, p, d);
    primal_vals[0] = primal_cur;
    
    for (iter = 0; iter < maxIter; iter ++)
    {
        loss_prev = loss_cur;

        calc_and_set_V(V, X, Z, B, n, p, d);
        calc_and_set_XTV(XTV, X, V, n, p);
        update_A(A, Atilde, UA, A_NZ, XTV, A_LHSinv, rho, n, p);
        update_Atilde(Atilde, A, UA, A_NZ, w, lambda, xi, y, tau, rho, mu_A, p);

        calc_and_set_W(W, X, A, n, p);
        calc_and_set_ZTW(ZTW, Z, W, n, p, d);
        update_B(B, Btilde, UB, ZTW, B_LHSinv, rho, n, p, d);
        update_Btilde(Btilde, B, UB, rho, mu_B, p, d);
        
        update_lambda(lambda, A, Atilde, w, xi, y, M, tau, p);
        update_xi(xi, Atilde, w, lambda, y, tau, p);

        primal_cur = obj_primal(X, Z, w, A, Atilde, B, Btilde, xi, lambda, UA, UB, y, mu_A, mu_B, tau, rho, n, p, d);
        primal_vals[iter+1] = primal_cur;
        
        update_UA(UA, A, Atilde, p);
        update_UB(UB, B, Btilde, p, d);
        update_y(y, A, Atilde, w, xi, lambda, tau, p);
        
        loss_cur = l2loss(X, Z, A, B, n, p, d);
        delta_loss = loss_cur - loss_prev;
        l2loss_vals[iter+1] = loss_cur;
        
        delta_AAtilde = 0.0;
        for (int i=0; i<p*p; i++)
            delta_AAtilde += (A[i] - Atilde[i])*(A[i] - Atilde[i]);

        delta_BBtilde = 0.0;
        for (int i=0; i<p*p; i++)
            delta_BBtilde += (B[i] - Btilde[i])*(B[i] - Btilde[i]);

        if (loss_cur > 10000)
        {
            std::cout << ">> admm iter = " << iter << ", l2loss = " << std::to_string(loss_cur).substr(0,6) << " exceeds the pre-set explosion threshold at 10000; force break" << std::endl;
        
            break;
        }
        
        if (verbose > 0)
        {
            if (iter % verbose == 0)
                std::cout << ">> admm iter = " << iter << ", delta_loss = " << std::to_string(delta_loss).substr(0,8) << ", l2loss = " << std::to_string(loss_cur).substr(0,6) << ", primal = " << std::to_string(primal_cur).substr(0,6) << std::endl;
        }

        if (ABS(delta_loss) < tol)
        {
            converge = true;
            break;
        }
    }
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapse = std::chrono::duration_cast<std::chrono::seconds> (end - begin).count();

    if (converge == false)
    {
        std::cout << "*** admm fails to converge; time elapse = " << std::to_string(elapse).substr(0,4) << " seconds ***" << std::endl;
    }
    else
    {
        if (verbose > 0)
            std::cout << "*** admm converges at iteration = " << iter << "; time elapse = " << std::to_string(elapse).substr(0,4) << " seconds ***" << std::endl;
    }
    
    delete [] M;
    delete [] V;
    delete [] W;
    delete [] XTV;
    delete [] ZTW;
    delete [] lambda;
    delete [] xi;
    
    delete [] UA;
    delete [] UB;
    delete [] y;
    
    return converge;
}


}

