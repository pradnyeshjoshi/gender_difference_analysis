#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:43:48 2020

@author: prady
"""

import pandas as pd
import logging, os
from glob import glob
from textblob import TextBlob
from tqdm import tqdm
from flair.models import TextClassifier
from flair.data import Sentence

def load_reddit(path):
    df = pd.read_csv(path)
    df['full_text'] = df['body'].astype(str).fillna('')
    return df

if __name__=='__main__':
    
    reddit_dir = '../../rawdata/'
    logging.info("loading reddit dataset")
    reddit_list = [y for x in os.walk(reddit_dir) for y in glob(os.path.join(x[0], '*.csv')) if 'comment' in y]
    
    classifier = TextClassifier.load('en-sentiment')
    for i in range(len(reddit_list)):
        r = reddit_list[i]
        print(r)
        df_reddit = load_reddit(r)
        if 'Unnamed: 0' in df_reddit.columns:
            del df_reddit['Unnamed: 0']
        m, n = df_reddit.shape
        
        if 'textblob_wordcount' in df_reddit.columns:
            continue

        df_reddit['textblob_polarity'] = 0
        df_reddit['textblob_subjectivity'] = 0
        df_reddit['textblob_wordcount'] = 0
        
        df_reddit['flair_value'] = ''
        df_reddit['flair_score'] = 0
        
        for j in tqdm(range(m)):

            tb = TextBlob(df_reddit.loc[j, 'full_text'])
            df_reddit.loc[j, 'textblob_polarity'] = tb.sentiment.polarity
            df_reddit.loc[j, 'textblob_subjectivity'] = tb.sentiment.subjectivity
            df_reddit.loc[j, 'textblob_wordcount'] = len(tb.words)
            
            sentence = Sentence(df_reddit.loc[j, 'full_text'])
            classifier.predict(sentence)
            if len(sentence.labels)>0:
                df_reddit.loc[j, 'flair_value'] = sentence.labels[0].value
                df_reddit.loc[j, 'flair_score'] = sentence.labels[0].score
        
        del df_reddit['full_text']
        df_reddit.to_csv(r, index=False, index_label=False)