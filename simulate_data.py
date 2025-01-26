"""
Main script for generate synthetic data

Copyright 2023, Jiahe Lin and George Michailidis
All Rights Reserved

Lin and Michailidis assert copyright ownership of this code base and its derivative
works. This copyright statement should not be removed or edited.

-----do not edit anything above this line---
"""
import argparse
import datetime
import os
import sys
import shutil

import numpy as np
import pickle
import yaml

from utils import sVarGenerator
from utils.utils_logging import get_logger

logger = get_logger()

def main():
    
    args.datasets = args.datasets.split(',')
    if len(args.config_override):
        args.config = args.config_override
        del args.config_override
    else:
        args.config = os.path.join('configs', 'datasets.yaml')
    logger.info(f"datasets={args.datasets}; config file={args.config}")
    
    with open(args.config) as f:
        meta_config = yaml.safe_load(f)
    
    ## read out some default setting
    if not os.path.exists(meta_config['defaults']['data_folder']):
        os.makedirs(meta_config['defaults']['data_folder'])
        logger.info(f"folder {meta_config['defaults']['data_folder']} created")
        
    for data_str in args.datasets:
        
        log_path = f"{meta_config['defaults']['data_folder']}/log_{data_str}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        if not os.path.exists(log_path):
            logger.info(f"more detailed logs will be redicted to `{log_path}`")
        else:
            logger.warning(f"more detailed logs will overwrite existing log file `{log_path}`")
            
        sys.stdout = open(log_path, 'w')
        print(f"config file={args.config}", end='\n\n')
        
        config = meta_config['defaults'].copy()
        config.update(meta_config[data_str])
        
        ## initialize data generator
        data_generator = sVarGenerator(dim = config['p'],
                                       nlags = config['nlags'],
                                       A_sparsity = config['A_sparsity'],
                                       A_sigLow = config['A_sigLow'],
                                       A_sigHigh = config['A_sigHigh'],
                                       A_type = config.get('A_type','erdos-renyi'),
                                       A_permutation = config.get('A_permutation',False),
                                       B_sparsity = config['B_sparsity'],
                                       B_sigLow = config['B_sigLow'],
                                       B_sigHigh = config['B_sigHigh'],
                                       B_sigDecay = config['B_sigDecay'],
                                       B_targetSR = config['B_targetSR'])
        ## generate data
        meta_data = data_generator.generate_dataset(n=config['n_max'],
                                    sigma=config['sigma'],
                                    noise_type=config.get('noise_type','Gaussian'),
                                    number_of_replica=config['n_replica'],
                                    graph_info=None,
                                    use_seed=True,
                                    seed=config.get('seed',None),
                                    save=False,
                                    filepath=None,
                                    max_trials=10000,
                                    verbose_trials=False)
        ## save down
        folder_name = os.path.join(meta_config['defaults']['data_folder'], data_str)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        else:
            shutil.rmtree(folder_name)
            os.mkdir(folder_name)
        
        for key, value in meta_data.items():
            with open(os.path.join(folder_name, f'{key}.pickle'), 'wb') as handle:
                pickle.dump(value, handle, protocol = pickle.HIGHEST_PROTOCOL)
            print(f' > SAVED: {folder_name}/{key}.pickle')
        
        ## save the graph
        data_generator.draw_graph(meta_data['graph_info']['graph_A'],
                                  save_file = os.path.join(folder_name, 'graph_A.png'))
        ## save a sample data
        data_generator.draw_sample_x(meta_data['data'][0],
                                     draw_count = 4,
                                     save_file = os.path.join(folder_name, 'selected_x.png'))
        
        ## print config to log
        print(f'\n********** config used **********')
        print(config)

        sys.stdout.close()
        sys.stdout = sys.__stdout__
    
    logger.info("done")
    return 0

if __name__ == "__main__":
    
    logger.info(f"python={'.'.join(map(str,sys.version_info[:3]))}; np={np.__version__}")
    logger.info(f"CWD={os.getcwd()}")
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--datasets',
                        type=str,
                        help='ids for dataset to be generated, separated by comma',
                        default='ds1')
    parser.add_argument('--config_override',
                        type=str,
                        help='path for the overriding config file used for generating datasets, default to empty string',
                        default='')

    global args
    args = parser.parse_args()
    
    main()
    
