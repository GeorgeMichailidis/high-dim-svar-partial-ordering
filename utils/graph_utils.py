"""
Various graph-related utilities

Copyright 2023, Jiahe Lin, Huitian Lei and George Michailidis
All Rights Reserved

Lin, Lei and Michailidis assert copyright ownership of this code base and its derivative
works. This copyright statement should not be removed or edited.

-----do not edit anything above this line---
"""



import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class graphUtil():
    def __init__(self):
        pass
    
    def _get_src_dst(self, A):
        ## count the src, get the destination
        src_counter, dst = {}, {}
        for i in range(len(A)):
            src_counter[i] = len(np.where(np.squeeze(A[i,:])!=0)[0])
            dst[i] = list(np.where(np.squeeze(A[:,i])!=0)[0])
        return src_counter, dst
        
    def topological_sort(self, A):
    
        src_counter, dst = self._get_src_dst(A)
        nodes_sorted, stack = [], [node for node, count in src_counter.items() if count==0]

        while stack:
            node = stack.pop(0)
            nodes_sorted.append(node)
            for child in dst[node]:
                src_counter[child] -= 1  ## remove the edge
                if src_counter[child] == 0:
                    stack.append(child)
        
        if all([count==0 for _, count in src_counter.items()]):
            return nodes_sorted
        else:
            print(f'Unable to perform topological sort!')
            return []
        
    def is_cyclic(self, A):
    
        _, graph = self._get_src_dst(A)
        visited = [0] * A.shape[0]
        
        def _check(node_id):
            if visited[node_id] == -1: return True
            if visited[node_id] == 1: return False
            visited[node_id] = -1
            for neighbor in graph[node_id]:
                if _check(neighbor):
                    return True
            visited[node_id] = 1
            return False
        
        for i in range(A.shape[0]):
            if not visited[i] and _check(i):
                return True
            
        return False
    
    def find_cycles(self, A):
        G = nx.from_numpy_array(A.transpose(),create_using=nx.DiGraph)
        try:
            cycles = list(nx.find_cycle(G,orientation='original'))
        except:
            cycles = []
        return cycles
    
    def is_cyclic_ij(self, A, i, j):
        
        A_new = copy.deepcopy(A)
        A_new[i,j] = 1
        
        return self.is_cyclic(A_new)

class graphGenerator():

    def __init__(self, seed_in_use=None):
        self.seed_in_use = seed_in_use ## placeholder
        self.is_cyclic = graphUtil().is_cyclic
        self.prior_pcts = [0.10, 0.20, 0.30, 0.50] ## the sequence of percentage for the correctly-specified zeros
        self.contam_pct = 0.10 ## the percentage for the mis-specified true support
        
    def gen_VAR_coef(self,dim,nlags,sparsity,sigLow,sigHigh,sigDecay=1,targetSR=0.5):
        """
        generate the transition matrix for a stationary VAR(q) process
        subject to targetSR (only approximate for nlags > 1)
        Return: p*p*q tensor
        """
        if nlags == 1:
            assert isinstance(sparsity,float)
            B = self._gen_VAR1_coef(dim, sparsity, targetSR, sig_range = (sigLow,sigHigh), diag=False)
            B = B[:,:,np.newaxis]
        else:
            assert (isinstance(sparsity, list) or isinstance(sparsity, tuple)) and len(sparsity) == nlags, f'incorrect specification for the sparsity of B'
            B = self._gen_VARq_coef(dim, q=nlags, sparsity=sparsity, targetSR=targetSR, diag=False, sig_ranges = [(sigLow*(sigDecay**i),sigHigh*(sigDecay**i)) for i in range(nlags)])
        return B
            
    def gen_structural_coef(self,dim,graph_type,sparsity,sigLow,sigHigh,permutation=True):
        """
        generate the adjacency matrix for a directed acyclic graph
        """
        
        if graph_type == 'erdos-renyi':
            skeleton = np.zeros((dim, dim))
            ## upper triangular
            for j in range(dim-1):
                skeleton[(j+1):,j] = np.random.binomial(1,sparsity,size=(dim-j-1,))
        elif graph_type == 'barabasi-albert':
            m = int(round(sparsity*(dim-1)/2))
            G0 = nx.barabasi_albert_graph(dim, m, seed=None, initial_graph=None)
            skeleton = np.triu(nx.to_numpy_array(G0),k=1).transpose()
        elif graph_type.startswith('chain'):
            offdiag_count = int(graph_type.split('-')[1])
            skeleton = np.zeros((dim,dim))
            for j in range(dim):
                for k in range(j+1,min(j+offdiag_count+1,dim)):
                    skeleton[k,j] = 1
        else:
            raise ValueError('unsupported graph_type')
            
        assert self.is_cyclic(skeleton) == False, 'cycles detected'
        assert np.all(np.diag(skeleton)==0)
        
        if permutation:
            pmt = np.random.permutation(np.eye(dim))
            skeleton = pmt.T.dot(skeleton).dot(pmt)
        
        ## add magnitude
        A = skeleton * np.random.uniform(sigLow,sigHigh,size=(dim,dim)) * np.random.choice([-1,1],size=(dim,dim))
        graph_A = nx.from_numpy_array(A.transpose(), create_using=nx.DiGraph)
        
        return A, graph_A
    
    def gen_A_prior_clean_dirty(self, A, prior_type = 'layered'):
        """
        generate a sequence of clean and dirty priors for A, where the prior is specified through (possibly) nonzeros entries known as a prior
        """
        assert prior_type in ['random','layered']
        
        p = A.shape[0]
        
        zero_set = list(zip(list(np.where(np.abs(A)<=1e-4)[0]), list(np.where(np.abs(A)<=1e-4)[1])))
        zero_set_nondiag = [(x,y) for x, y in zero_set if x != y]
        zero_count = len(zero_set_nondiag)
        
        support_set = list(zip(list(np.where(np.abs(A)>1e-4)[0]), list(np.where(np.abs(A)>1e-4)[1])))
        support_count = len(support_set)
        
        contam_index_set = np.random.choice(support_count, size = int(self.contam_pct * support_count))
        contam_coordinates = [coordinate for idx, coordinate in enumerate(support_set) if idx in contam_index_set]
        
        A_NZ_clean, A_NZ_dirty = {}, {}
        for pct in self.prior_pcts:
        
            if prior_type == 'layered':
                h, target = 1, pct * zero_count
                while h <= p:
                    prior_zeros_count = (p+(p-h+1))*h/2 - h
                    if prior_zeros_count >= target:
                        print(f' ** target_prior_pct={pct}; h={h}; eff_prior_pct={prior_zeros_count/zero_count*100:.2f}%')
                        break
                    h += 1
                zero_coordinates = [(i,j) for i in range(h) for j in range(i+1,p)]
            elif prior_type == 'random':
                zero_index_set = np.random.choice(zero_count, size = int(pct * zero_count))
                zero_coordinates = [coordinate for idx, coordinate in enumerate(zero_set_nondiag) if idx in zero_index_set]
            
            assert set(zero_coordinates).intersection(set(contam_coordinates)) == set()
            mtx_clean, mtx_dirty = 1 - np.identity(p,dtype=np.int32), 1 - np.identity(p,dtype=np.int32)
            for i in range(p):
                for j in range(p):
                    if (i,j) in zero_coordinates:
                        mtx_clean[i,j] = 0
                        mtx_dirty[i,j] = 0
                    elif (i,j) in contam_coordinates:
                        mtx_dirty[i,j] = 0
            
            A_NZ_clean[pct] = mtx_clean
            A_NZ_dirty[pct] = mtx_dirty
            
        return A_NZ_clean, A_NZ_dirty
        
    def draw_graph(self, G, save_file = None):
        
        n_nodes = len(G.nodes())
        avg_degree = sum([v for _, v in G.degree()])/n_nodes
        
        plt.rcParams["figure.figsize"] = (10,10)

        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color="tab:blue", edgecolors="tab:gray",alpha=0.3)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color="tab:red", alpha=0.5, arrows = True)
    
        plt.axis("off")
        plt.title(f'n_nodes = {n_nodes}, avg_degree = {avg_degree:.2f}')
        
        if save_file is not None:
            plt.savefig(save_file,facecolor='w')
            plt.close()
        else:
            plt.show()
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
        
    def calc_avg_degree(self,mtx,type='intra'):
        p = mtx.shape[0]
        assert mtx.shape[0] == mtx.shape[1]
        if type == 'intra':
            return 2*np.sum(np.abs(mtx)>0)/p
        elif type == 'inter':
            return np.sum(np.abs(mtx)>0)/p
        else:
            raise ValueError('invalid type')
        
    def _gen_randmtx(self, m, n, sparsity = None, sig_range=(1,3), diag=False):
        """ generate a m by n random matrix with sparsity """
        if sparsity is None:
            mtx = np.random.uniform(sig_range[0],sig_range[1],size=(m,n))
        else:
            mtx = np.random.uniform(sig_range[0],sig_range[1],size=(m,n))*np.random.binomial(1,sparsity,size=(m,n))
        
        mtx = mtx * np.random.choice([-1,1],size=(m,n))
        if diag:
            mtx = np.diag(np.diag(mtx))
        return mtx

    def _get_sr(self, mtx):
        """ obtain the spectral radius of a square matrix mtx """
        return max(np.abs(np.linalg.eigvals(mtx)))

    def _companion_stack(self, list_of_mtx, verbose=False):
        """ put a list of matrix for lags into the companion form """
        num_lags = len(list_of_mtx)
        
        if num_lags == 1:
            if verbose:
                print('[companion stack] WARNING: num_lags = 1, degenerate companion form.')
            return list_of_mtx[0]
        
        p = list_of_mtx[0].shape[0]
        identity = np.diag(np.ones(((num_lags-1)*p,))) ## identity matrix of size (num_lags-1)*p
        zeros = np.zeros(((num_lags-1)*p,p))
        bottom = np.concatenate([identity, zeros],axis=1)
        top = np.concatenate(list_of_mtx,axis=1)
        
        return np.concatenate([top,bottom],axis=0)

    def _companion_disagg(self, mtx, num_lags, verbose=False):
        """ extract the lag coefficients from a companion form and put them into a list """
        if mtx.shape[1] == num_lags:
            if verbose:
                print('[companion disaggregate] nothing to disaggregate')
            return [mtx]
        
        list_of_mtx = []
        assert mtx.shape[1] % num_lags == 0, 'number of columns in the companion matrix is not a multiple of the number of lags; something went wrong'
        p = int(mtx.shape[1]//num_lags)
        for i in range(num_lags):
            list_of_mtx.append(mtx[:p,(i*p):((i+1)*p)])
        return list_of_mtx

    def _scale_coefs(self, list_of_mtx, target, verbose=False):
        """ scale the matrix so that its spectral radius is smaller than the target """
        num_lags = len(list_of_mtx)
        mtx_comp = self._companion_stack(list_of_mtx)
        old_sr = self._get_sr(mtx_comp)
        mtx_comp_scaled = target / old_sr * mtx_comp
        
        list_of_mtx_new = self._companion_disagg(mtx_comp_scaled, num_lags)
        ## since the scaling won't be exact, do some sanity checks
        mtx_comp_new = self._companion_stack(list_of_mtx_new)
        new_sr = self._get_sr(mtx_comp_new)
        if verbose:
            print(f'spectral radius before scaling = {old_sr:.3f}, after scaling = {new_sr:.3f}')
        if new_sr >= 1:
            print(f'![WARNING]: spectral radius after scaling = {new_sr:.3f}')
        return list_of_mtx_new

    def _gen_VAR1_coef(self, p, sparsity, targetSR=0.5, sig_range=(0.5,1), diag = False):
        """
        generate the transition matrix for a stationary VAR(1) process
        Return: p*p matrix
        """
        raw_mtx = self._gen_randmtx(p, p, sparsity, sig_range=sig_range, diag=diag)
        scaled_mtx = self._scale_coefs([raw_mtx],target=targetSR)[0]
        return scaled_mtx

    def _gen_VARq_coef(self, p, q, sparsity, targetSR = 0.8, diag= False, sig_ranges = None):
        """
        generate the transition matrix for a stationary VAR(q) process
        Note that the targetSR will not be attained exactly
        Return: p*p*q tensor
        """
        if not sig_ranges:
            sig_ranges = [[2*i,2*i+1] for i in range(q)]
            sig_ranges.reverse()
        
        raw_mtx_list = []
        for i in range(q):
            raw_mtx = self._gen_randmtx(p,p,sparsity[i],sig_range=sig_ranges[i],diag=diag)
            raw_mtx_list.append(raw_mtx)
        
        scaled_mtx_list = self._scale_coefs(raw_mtx_list, target=targetSR)
        return np.stack(scaled_mtx_list,axis=-1)
