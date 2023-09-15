
"""
Main script for performing model fitting on synthetic data

Copyright 2023, Jiahe Lin, Huitian Lei and George Michailidis
All Rights Reserved

Lin, Lei and Michailidis assert copyright ownership of this code base and its derivative
works. This copyright statement should not be removed or edited.

-----do not edit anything above this line---
"""


import os
import sys

print(f'python version={".".join(map(str,sys.version_info[:3]))}')
print(f'current working dir={os.getcwd()}')

import yaml
import importlib
import argparse
import pickle
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from utils import Evaluator

###################################
_PRIORKEYs = [] #[0.10, 0.20, 0.50]
###################################
    
parser = argparse.ArgumentParser(description='')
parser.add_argument('--ds_str', type=str, help='dataset to run',default='ds1')
parser.add_argument('--replica_id', type=int, help='replica id',default=0)
parser.add_argument('--train_size', type=int, help='sample size used for model training',default=200)
parser.add_argument('--standardize',help='whether to standardize the data',action='store_true')
parser.add_argument('--report',help='whether to report metrics',action='store_true')

def main():
    
    global args
    args = parser.parse_args()
    
    _CONFIG = os.path.join('configs',f'{args.ds_str}.yaml')
    
    config_key = f'{args.ds_str}-{args.train_size}' + ('' if not args.standardize else '-standardize')
    print(f'===========================')
    print(f'* ds_str={args.ds_str}; train_size={args.train_size}; standardize={args.standardize}; config_file={_CONFIG}, config_key={config_key}')
    print(f'===========================')

    with open(_CONFIG) as f:
        meta_config = yaml.safe_load(f)
    
    assert config_key in meta_config, f'{config_key} missing from config file {_CONFIG}'
    
    args.configs = meta_config['default'].copy()
    args.configs.update(meta_config[config_key])
    
    models = importlib.import_module('src')
    svarClass = getattr(models,args.configs['model_class'])
    svar = svarClass(tau = args.configs['tau'],
                    rho = args.configs['rho'],
                    max_admm_iter = args.configs['max_admm_iter'],
                    admm_tol = args.configs['admm_tol'],
                    verbose = 50,
                    tol = args.configs['tol'],
                    max_epoch = args.configs['max_epoch'],
                    threshold_A = args.configs['threshold_A'],
                    threshold_B = args.configs['threshold_B'],
                    SILENCE = False)

    with open(f'data/sim/{args.ds_str}/graph_info.pickle','rb') as handle:
        graph_info = pickle.load(handle)
    with open(f'data/sim/{args.ds_str}/data.pickle','rb') as handle:
        data = pickle.load(handle)
        
    xdata = data[args.replica_id][:args.train_size]
    if args.standardize:
        scaler = StandardScaler()
        xdata = scaler.fit_transform(xdata)
    
    for prior_key in [0.0] + _PRIORKEYs:
    
        print('################')
        if prior_key == 0.0:
            print(f'## no priors')
            A_NZ = None
        else:
            print(f'## {prior_key*100:.0f}% priors')
            A_NZ = graph_info['prior_clean'][prior_key]
        print('################')
        
        out = svar.fitSVAR(xdata,
                           q=args.configs['nlags'],
                           mu_A=args.configs['mu_A'],
                           mu_B=args.configs['mu_B'],
                           mu_B_refit=args.configs.get('mu_B_refit',None),
                           A_NZ=A_NZ)
        
        x_forecast = svar.forecast(xdata, out['A'], out['B'], horizon=1)
        if args.standardize:
            x_forecast = scaler.inverse_transform(x_forecast)
        x_forecast_actual = np.expand_dims(data[args.replica_id][args.train_size], axis=0)
            
        if args.report:
                
            reports_skeleton = get_skeleton_report(graph_info, out)
            print(reports_skeleton)
            
            reports_x_forecast = get_x_report_forecast(x_forecast_actual, x_forecast)
            print(reports_x_forecast)
            
    return 0

def get_skeleton_report(graph_info, out):
    
    evaluator = Evaluator()
    reports_skeleton = []
    
    ## A
    report = evaluator.report(graph_info['A'], out['A'])
    report['key'] = 'A'
    reports_skeleton.append(report)
    
    ## B
    for lag_id in range(graph_info['B'].shape[-1]):
        report = evaluator.report(graph_info['B'][:,:,lag_id], out['B'][:,:,lag_id])
        report['key'] = f'B_{lag_id+1}'
        reports_skeleton.append(report)
    
    return reports_skeleton
    
def get_x_report_forecast(x_forecast_actual, x_forecast):
    
    assert x_forecast_actual.shape[0] == 1
    
    rmse = mean_squared_error(x_forecast_actual.reshape(-1,1), x_forecast.reshape(-1,1), squared=False)/np.linalg.norm(x_forecast_actual.reshape(-1,1))
    return {'forecast_l2': round(rmse,3)}

if __name__ == "__main__":
    main()
