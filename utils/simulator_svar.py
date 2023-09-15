"""
Generator for a structural VAR model used in synthetic data experiments

Copyright 2023, Jiahe Lin, Huitian Lei and George Michailidis
All Rights Reserved

Lin, Lei and Michailidis assert copyright ownership of this code base and its derivative
works. This copyright statement should not be removed or edited.

-----do not edit anything above this line---
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime

from .graph_utils import graphGenerator

class sVarGenerator(graphGenerator):
    
    def __init__(
        self,
        dim,
        nlags,
        A_sparsity = 0.05,
        A_sigLow = 0.25,
        A_sigHigh = 1,
        A_type = 'erdos-renyi',
        A_permutation = True,
        B_sparsity = 0.05,
        B_sigLow = 1,
        B_sigHigh = 3,
        B_sigDecay = 1,
        B_targetSR = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.nlags = nlags
        self.A_sparsity = A_sparsity
        self.A_sigLow = A_sigLow
        self.A_sigHigh = A_sigHigh
        self.A_type = A_type
        self.A_permutation = A_permutation

        if nlags > 1:
            assert (isinstance(B_sparsity, list) or isinstance(B_sparsity, tuple)) and len(B_sparsity) == nlags, f'incorrect specification for the sparsity of B'
        self.B_sparsity = B_sparsity
        self.B_sigLow = B_sigLow
        self.B_sigHigh = B_sigHigh
        self.B_sigDecay = B_sigDecay
        self.B_targetSR = B_targetSR
        self.seed_pool = np.random.choice(np.arange(0,2**15),10000)

    def gather_params(self):
        self._params = {}
        for attr_name in ['dim', 'nlags', 'A_sparsity','A_sigLow', 'A_sigHigh', 'A_type', 'B_sparsity', 'B_sigLow', 'B_sigHigh', 'B_sigDecay', 'B_targetSR','seed_in_use']:
            self._params[attr_name] = getattr(self, attr_name)

    def _check_stationarity(self, A, B):

        I_mins_A_inv, AinvB = np.linalg.inv(np.identity(A.shape[1])-A), []
        for i in range(B.shape[-1]):
            AinvB.append(np.matmul(I_mins_A_inv, B[:,:,i]))

        mtx_comp = self._companion_stack(AinvB)
        sr = self._get_sr(mtx_comp)
        is_stationary = False if sr > 0.99 else True
        return {'is_stationary': is_stationary, 'sr': sr}

    def generate_graph(self, use_seed=False, seed=None, max_trials=10000, verbose_trials=False, get_priors=True):

        if use_seed and seed is not None:
            self.seed_in_use = seed
            np.random.seed(seed)
            max_trials = 1
            use_random_seed = False
        else:
            use_random_seed = True

        for trial in range(max_trials):

            if use_seed and use_random_seed:
                seed = self.seed_pool[trial]
                self.seed_in_use = seed
                np.random.seed(seed)

            A, graph_A = self.gen_structural_coef(self.dim,self.A_type,self.A_sparsity,self.A_sigLow,self.A_sigHigh,self.A_permutation)
            B = self.gen_VAR_coef(self.dim,self.nlags,self.B_sparsity,self.B_sigLow,self.B_sigHigh,self.B_sigDecay,self.B_targetSR)

            stationarity_check = self._check_stationarity(A,B)
            if verbose_trials > 0:
                if trial % verbose_trials == 0:
                    print(f'trial_id={trial}, spectral_radius={stationarity_check["sr"]:.3f}, seed_in_use={seed}')
            if stationarity_check['is_stationary']:
                break

        if trial == max_trials - 1 and not stationarity_check['is_stationary']:
            warnings.warn(f'no valid coef found within {max_trials}')

        graph_info = {'A': A, 'B': B, 'graph_A': graph_A, 'seed': seed}


        print(f'[{datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")}] Graph Generation Completed; seed={self.seed_in_use}')
        print(f' > trial_id={trial}, reduced VAR spectral_radius={stationarity_check["sr"]:.2f}')
        print(f' > avg_degrees: A={self.calc_avg_degree(A,type="intra"):.2f}; ', end='')
        for i in range(B.shape[-1]):
            end = '\n' if i == B.shape[-1]-1 else ''
            print(f'B_{i+1}={self.calc_avg_degree(B[:,:,i],type="inter"):.2f}; ', end=end)
        if get_priors:
            prior_clean, prior_dirty = self.gen_A_prior_clean_dirty(A)
            graph_info['prior_clean'] = prior_clean
            graph_info['prior_dirty'] = prior_dirty
            print(f' > prior_clean/dirty: {list(prior_clean.keys())}')
        return graph_info

    def draw_sample_x(self, xdata, draw_count = 4, save_file = None):

        p = xdata.shape[1]
        coordinates_to_draw = np.random.choice(p, size=draw_count)
        fig, axs = plt.subplots(2,draw_count//2, figsize=(20,8),constrained_layout=True)
        for idx, coord in enumerate(coordinates_to_draw):
            fig_id = idx%2, idx//2
            axs[fig_id].plot(np.arange(xdata.shape[0]),xdata[:,coord])
            axs[fig_id].set_title(f'x_i = {coord}')
        if save_file is not None:
            fig.savefig(save_file,facecolor='w')
            plt.close()
        else:
            plt.show()

    def gen_svar_data(self,n,A,B,sigma,noise_type='Gaussian',burn_in=200):

        p, q = A.shape[0], B.shape[-1]
        I_minus_A_inv = np.linalg.inv(np.identity(p) - A)

        xdata = np.zeros((n+burn_in,p))

        if noise_type == 'Gaussian':
            edata = np.random.normal(loc=0,scale=1.0,size=(n+burn_in,p))
        elif noise_type == 'Laplace':
            edata = 1.0/np.sqrt(2) * np.random.laplace(loc=0,scale=1.0,size=(n+burn_in,p))
        elif noise_type == 'T4':
            edata = 1.0/np.sqrt(2) * np.random.standard_t(df=4, size=(n+burn_in,p))
        else:
            raise ValueError('unrecognized noise_type')

        if isinstance(sigma, list) or isinstance(sigma, tuple):
            sigmas = np.diag(sorted(np.random.uniform(sigma[0],sigma[1],size=p)))
            edata = np.matmul(edata, sigmas)
        else:
            edata = sigma * edata

        for t in range(burn_in):
            xdata[t,:] = edata[t,:]
        for t in range(burn_in, n + burn_in):
            RHS = edata[t,:]
            for lag_id in range(1,q+1):
                RHS += np.dot(B[:,:,lag_id-1], xdata[t-lag_id,:].T)
            xdata[t,:] = np.dot(I_minus_A_inv,RHS)

        return xdata[burn_in:,:]

    def generate_dataset_from_graph(self, graph_info, n, sigma, noise_type, number_of_replica, use_seed=True, seed=None):

        if use_seed:
            if seed is None:
                seed = graph_info['seed']
            np.random.seed(seed)

        A, B = graph_info['A'], graph_info['B']

        data_with_replica = {}
        for i in range(number_of_replica):
            data_with_replica[i] = self.gen_svar_data(n, A, B, sigma, noise_type, burn_in=200)

        print(f'[{datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")}] Data Generation Completed ({number_of_replica} replicas); seed={seed}')
        return data_with_replica

    def generate_dataset(self, n, sigma, noise_type, number_of_replica, graph_info = None, use_seed = True, seed = None, save = False, filepath = None, max_trials = 10000, verbose_trials = False):
        
        if graph_info is not None:
            with open(graph_info,"rb") as handle:
                graph_info = pickle.load(handle)
        else:
            graph_info = self.generate_graph(use_seed = use_seed, seed = seed, max_trials = max_trials, verbose_trials = verbose_trials)
        data_with_replica = self.generate_dataset_from_graph(graph_info, n, sigma, noise_type, number_of_replica, use_seed = use_seed, seed = seed)

        self.gather_params()
        meta_data = {'graph_info': graph_info, 'data': data_with_replica, 'params': self._params}

        if save:
            if not os.path.exists('data/sim'):
                os.mkdir('data/sim')
            if filepath is None:
                filepath = f'data/sim/SVAR_{self.A_type}_{datetime.datetime.now().strftime("%Y%m%d")}_p{self.dim}n{n}.pickle'
            file_prefix = filepath.replace('.pickle','')
            ## save the data
            with open(filepath, 'wb') as handle:
                pickle.dump(meta_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
            ## save the graph
            self.draw_graph(graph_info['graph_A'], save_file = f'{file_prefix}_graph_A.png')
            ## save a sample data
            self.draw_sample_x(data_with_replica[0], draw_count = 4, save_file = f'{file_prefix}_selected_coordinates.png')
            print("============")
            print(f"data saved; to retrieve, use the following command:")
            print(f"with open('{filepath}','rb') as handle:")
            print(f"    meta_data = pickle.load(handle)")
            print("============")

        return meta_data
