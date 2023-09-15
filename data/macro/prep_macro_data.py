"""
python prep_macro_data.py --vintage='202209'
"""

import os
import sys
import argparse
import yaml
import pickle
import wget

import pandas as pd
import numpy as np
import datetime
from scipy import stats
import collections

parser = argparse.ArgumentParser(description='')
parser.add_argument('--vintage', type=str, help='vintage', default='202209')

def main():
    
    global args
    args = parser.parse_args()
    
    setattr(args, 'first_data_date', pd.to_datetime("1973-03-31"))
    args.winsorize = None
    
    filepath = f'data/macro/{args.vintage}_Qraw.csv'
    if not os.path.exists(filepath):
        url = 'https://files.stlouisfed.org/files/htdocs/fred-md/quarterly/current.csv'
        new_filepath = f'data/macro/{datetime.datetime.now().strftime("%Y%m")}_Qraw.csv'
        filepath_downloaded = wget.download(url,new_filepath)
        print(f' >> {filepath} does not exists. Downloaded the latest from FRED-QD and saved as {new_filepath}')
        setattr(args, 'vintage', datetime.datetime.now().strftime("%Y%m"))
        filepath = new_filepath
    
    ## read in raw quarterly data
    QD_raw = pd.read_csv(filepath)
    QD_raw = QD_raw.iloc[2:,:].copy()
    QD_raw['sasdate'] = pd.to_datetime(QD_raw['sasdate'],format='%m/%d/%Y')
    QD_raw['sasdate'] = [date - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd() for date in QD_raw['sasdate']]
    QD_raw.rename(columns={'sasdate':'date'},inplace=True)
    QD_raw = QD_raw.set_index('date').dropna(how='all')
    print(f"Stage: read in raw data; last data date in QD_raw = {QD_raw.index[-1].strftime('%Y-%m-%d')}")
         
    ## prepare variable and their respective group information
    datadict = pd.read_excel(f'data/macro/dictionary.xlsx',sheet_name='SVAR')
    datadict = datadict.loc[datadict['SVAR_GROUP'].notna()].sort_values(by=['SVAR_GROUP','MNEMONIC'])
    
    variables = datadict[['MNEMONIC','DESCRIPTION','TCODE','SVAR_GROUP']].set_index('MNEMONIC').to_dict('index') # dict(zip(datadict['MNEMONIC'], datadict['DESCRIPTION']))
    assert list(variables.keys()) == list(datadict['MNEMONIC'])
    group_info = datadict[['SVAR_GROUP','MNEMONIC']]

    QD = prep_data(QD_raw, variables)
    A_NZ, df_by_group = prep_prior(group_info)
    
    meta_data = {'df': QD,
                 'A_NZ': A_NZ,
                 'group_info': dict(zip(df_by_group['SVAR_GROUP'],df_by_group['MNEMONIC'])),
                 'description': dict(zip(datadict['MNEMONIC'],datadict['DESCRIPTION']))
                }
    out_filepath = f'data/macro/{args.vintage}.pickle'
    
    with open(out_filepath, 'wb') as handle:
        pickle.dump(meta_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    print(f'[FILE SAVED {datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")}] {out_filepath}; keys={list(meta_data.keys())}')
    return 0
    
def prep_data(QD_raw, variables):

    print('Stage: checking missing values')
    ## I: filter to required columns, check missing value & prep for imputation
    QD = QD_raw.loc[QD_raw.index >= args.first_data_date - pd.tseries.offsets.DateOffset(days=367) + pd.tseries.offsets.QuarterEnd(), list(variables.keys())].copy()

    variables_needs_imputation = {}
    for col in QD.columns.to_list():
        if any(QD[col].isna()):
            missing_dates = QD.index[QD[col].isna()].to_list()
            missing_dates_filtered = list(filter(lambda x: x >= args.first_data_date, missing_dates))
            if len(missing_dates_filtered):
                variables_needs_imputation[col] = missing_dates_filtered
                print(f' >> {col} ({variables[col]["DESCRIPTION"]}) has NA values on {[x.strftime("%Y-%m-%d") for x in missing_dates_filtered]} and will be imputed')

    ## II: handle missing values
    if len(variables_needs_imputation):
        print('Stage: impute missing values')
        for col, missing_dates in variables_needs_imputation.items():
            if any(QD[col].isna()):
                QD[col].ffill(inplace=True)
                print(f' >> value for {col} has been forward-filled')
    else:
        print(' >> no need for missing values imputation')
    Qlevels = QD.copy()

    ## III: applying transf
    print('Stage: applying transformation and touch up')
    for col in QD.columns:
        QD[col] = apply_tcode(QD[col],variables[col]['TCODE'],freq=4)

    ## IV: filtering, winsorizing, checking
    QD = QD[QD.index >= args.first_data_date]
    if args.winsorize is not None:
        for col in QD.columns:
             QD[col] = stats.mstats.winsorize(QD[col], limits=args.winsorize)
    for col in QD.columns.to_list():
        if any(QD[col].isna()):
            print(f'!! {col} has NA values on {QD.index[QD[col].isna()]}; this should not happen')
    
    print(f" >> first_data_date = {QD.index[0].strftime('%Y-%m-%d')}; last_data_date = {QD.index[-1].strftime('%Y-%m-%d')}; QD.shape = {QD.shape}")
    return QD

def prep_prior(group_info):
    
    print('Stage: prep priors; ', end='')
    
    n = group_info.shape[0]
    counter = list(collections.Counter(group_info['SVAR_GROUP']).values())
    boundaries = np.cumsum(counter)
    
    A_NZ = np.zeros((n,n))
    for i, group_end in enumerate(boundaries):
        group_start = 0 if i == 0 else boundaries[i-1]
        A_NZ[group_start:group_end, :group_end] = 1
    for i in range(n):
        A_NZ[i,i] = 0
    print(f'A_NZ.shape = {A_NZ.shape}')
    
    df_by_group = group_info.groupby('SVAR_GROUP',as_index=False).agg({'MNEMONIC':list})
    return A_NZ, df_by_group

def apply_tcode(x,tcode,freq=4):

    ## x: pd.Series -> the raw time series to be processed
    ## tcode: int -> transformation code; 1-7 are what FRED db used; 8- are customized
    if tcode == 1:
        ret = x
    elif tcode == 2: # diff
        ret = x.diff()
    elif tcode == 3: ## twice diff
        dx = x.diff()
        ret = dx.diff()
    elif tcode == 4: ## log
        ret = np.log(x)
    elif tcode == 5: ## 100\delta log (approx for pct change)
        logx = np.log(x)
        ret = 100 * logx.diff()
    elif tcode == 6: ## twice delta log (change in growth rate)
        logx = np.log(x)
        dlogx = logx.diff()
        ret = 100 * dlogx.diff()
    elif tcode == 7: ## diff of pct change (change in growth rate)
        pct_change = x.pct_change()
        ret = 100 * pct_change.diff()
    #### customized
    elif tcode == 8: ## QoQ AR
        pct_change = x.pct_change()
        ret = 100 * ((1+pct_change)**4-1)
    elif tcode == 9: ## YoY
        ret = 100 * (x.pct_change(freq))
    elif tcode == 10: ## divided by 10
        ret = x / 10.0
    elif tcode == 11: ## divided by 100
        ret = x / 100.0
    elif tcode == 12: ## 100*log
        ret = 100 * np.log(x)
    elif tcode == 13: ## 100*x
        ret = 100*x
    else:
        raise ValueError(f'tcode = {tcode}, unrecognized')
    return ret


    
if __name__ == "__main__":
    main()
    
