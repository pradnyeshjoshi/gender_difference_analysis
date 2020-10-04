#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:43:48 2020

@author: prady
"""

import pandas as pd
import logging, os
from glob import glob
from tqdm import tqdm
import moralstrength as ms

def load_reddit(path):
    df = pd.read_csv(path)
    df['full_text'] = df['body'].astype(str).fillna('')
    return df

if __name__=='__main__':
    
    reddit_dir = '../../rawdata/'
    logging.info("loading reddit dataset")
    reddit_list = [y for x in os.walk(reddit_dir) for y in glob(os.path.join(x[0], '*.csv')) if 'comment' in y]

    print(len(reddit_list))
    for i, r in enumerate(reddit_list):

        print(i, r)
        df_reddit = load_reddit(r)
        
        if 'care' in df_reddit.columns:
            continue
        
        if 'Unnamed: 0' in df_reddit.columns:
            del df_reddit['Unnamed: 0']
        m, n = df_reddit.shape

        df_reddit['care'] = 0
        df_reddit['fairness'] = 0
        df_reddit['loyalty'] = 0
        df_reddit['authority'] = 0
        df_reddit['purity'] = 0
        df_reddit['non_moral'] = 0
        
        for j in tqdm(range(m)):

            d = ms.string_moral_values(df_reddit.loc[j, 'full_text'])
            df_reddit.loc[j, 'care'] = d['care']
            df_reddit.loc[j, 'fairness'] = d['fairness']
            df_reddit.loc[j, 'loyalty'] = d['loyalty']
            df_reddit.loc[j, 'authority'] = d['authority']
            df_reddit.loc[j, 'purity'] = d['purity']
            df_reddit.loc[j, 'non_moral'] = d['non-moral']
        
        del df_reddit['full_text']
        df_reddit.to_csv(r, index=False, index_label=False)