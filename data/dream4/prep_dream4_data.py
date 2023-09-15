
import os
import sys
import pickle
import warnings

import pandas as pd
import numpy as np
import datetime

_DATASETS = ['dream4_100_01', 'dream4_100_02', 'dream4_100_03', 'dream4_100_04', 'dream4_100_05']
_DIR = 'data/dream4'

def main():
    
    for dataset_key in _DATASETS:
    
        filename = os.path.join(_DIR, f'{dataset_key}.xlsx')
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = pd.read_excel(filename,sheet_name='data',index_col=0,engine="openpyxl")
            Atrue = pd.read_excel(filename,sheet_name='A',index_col=0,engine="openpyxl")
            df_src_tgt = pd.read_excel(filename,sheet_name='src_target',index_col=0,engine="openpyxl")
            
        ## Process prior information based on regulator and target
        src_raw, tgt_raw = set(list(df_src_tgt['Regulator'])), set(list(df_src_tgt['Target']))
        src, tgt = src_raw.difference(tgt_raw), tgt_raw.difference(src_raw)
        all_Gs = list(df.columns[:-1])
        uclr = set(all_Gs).difference(src).difference(tgt)
        
        ## create prior based on src and tgt
        A_NZ = np.ones((100,100))
        for i in range(100):
            if all_Gs[i] in src:
                A_NZ[i,:] = 0 ## no incoming
            elif all_Gs[i] in tgt:
                A_NZ[:,i] = 0 ## no outgoing
            A_NZ[i,i] = 0 ## no self-loop whatsoever
        A_NZ = pd.DataFrame(data=A_NZ, index=all_Gs, columns=all_Gs)
        
        meta_data = {'df': df, 'Atrue': Atrue, 'A_NZ': A_NZ, 'df_src_tgt': df_src_tgt}
        
        picklename = os.path.join(_DIR, f'{dataset_key}.pickle')
        with open(picklename, 'wb') as handle:
            pickle.dump(meta_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
        print(f'>> {picklename} saved.' )
        
    return 0


if __name__ == "__main__":
    main()
    
