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
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
from nrclex import NRCLex
from sentistrength import PySentiStr

def get_moralstrength(df):
    df['care'] = 0
    df['fairness'] = 0
    df['loyalty'] = 0
    df['authority'] = 0
    df['purity'] = 0
    df['non_moral'] = 0
    df.reset_index(drop=True, inplace=True)
    for j in tqdm(range(df.shape[0])):
        d = ms.string_moral_values(df.loc[j, 'text'])
        df.loc[j, 'care'] = d['care']
        df.loc[j, 'fairness'] = d['fairness']
        df.loc[j, 'loyalty'] = d['loyalty']
        df.loc[j, 'authority'] = d['authority']
        df.loc[j, 'purity'] = d['purity']
        df.loc[j, 'non_moral'] = d['non-moral']
    return df

def get_flair_sentiments(df_reddit):
    classifier = TextClassifier.load('en-sentiment')
    df_reddit['textblob_polarity'] = 0
    df_reddit['textblob_subjectivity'] = 0
    df_reddit['textblob_wordcount'] = 0
    df_reddit['flair_value'] = ''
    df_reddit['flair_score'] = 0
    for j in tqdm(range(df_reddit.shape[0])):
        tb = TextBlob(df_reddit.loc[j, 'text'])
        df_reddit.loc[j, 'textblob_polarity'] = tb.sentiment.polarity
        df_reddit.loc[j, 'textblob_subjectivity'] = tb.sentiment.subjectivity
        df_reddit.loc[j, 'textblob_wordcount'] = len(tb.words)
        sentence = Sentence(df_reddit.loc[j, 'text'])
        classifier.predict(sentence)
        if len(sentence.labels)>0:
            df_reddit.loc[j, 'flair_value'] = sentence.labels[0].value
            df_reddit.loc[j, 'flair_score'] = sentence.labels[0].score
    return df_reddit

def get_nrc_sentiments(df):
    df['fear'] = 0
    df['anger'] = 0
    df['anticip'] = 0
    df['trust'] = 0
    df['surprise'] = 0
    df['positive'] = 0
    df['negative'] = 0
    df['sadness'] = 0
    df['disgust'] = 0
    df['joy'] = 0
    df['text'] = df['text'].fillna("")
    df.reset_index(drop=True, inplace=True)
    for j in tqdm(range(df.shape[0])):
        d = NRCLex(df.loc[j, 'text']).affect_frequencies
        df.loc[j, 'fear'] = d['fear']
        df.loc[j, 'anger'] = d['anger']
        df.loc[j, 'anticip'] = d['anticip']
        df.loc[j, 'trust'] = d['trust']
        df.loc[j, 'surprise'] = d['surprise']
        df.loc[j, 'positive'] = d['positive']
        df.loc[j, 'negative'] = d['negative']
        df.loc[j, 'sadness'] = d['sadness']
        df.loc[j, 'disgust'] = d['disgust']
        df.loc[j, 'joy'] = d['joy']
    return df

def get_sentistrength(df):
    senti = PySentiStr()
    senti.setSentiStrengthPath('~/softwares/SentiStrengthCom.jar')
    senti.setSentiStrengthLanguageFolderPath('~/softwares/SentStrength_Data_Sept2011/')
    df["text"] = [t if t!="" else " " for t in df['text']]
    result = senti.getSentiment(df["text"], score='trinary')
    df["sentistrength_pos"] = [r[0] for r in result]
    df["sentistrength_neg"] = [r[1] for r in result]
    df["sentistrength_neutral"] = [r[2] for r in result]
    return df

def get_wordcount(df, file_type="comment"):
    if file_type == "comment":
        df['comment_wordcount'] = 0
        for j in tqdm(range(df.shape[0])):
            tb = TextBlob(df_reddit.loc[j, 'text'])
            df_reddit.loc[j, 'comment_wordcount'] = len(tb.words)
    else:
        df['title_wordcount'] = 0
        df['post_wordcount'] = 0
        df['title'] = df['title'].astype(str)
        df['selftext'] = df['selftext'].astype(str)
        for j in tqdm(range(df.shape[0])):
            tb = TextBlob(df_reddit.loc[j, 'title'])
            df_reddit.loc[j, 'title_wordcount'] = len(tb.words)
            tb = TextBlob(df_reddit.loc[j, 'selftext'])
            df_reddit.loc[j, 'post_wordcount'] = len(tb.words)
    return df

if __name__ == '__main__':

    reddit_dir = '../data/rawdata/'
    logging.info("loading reddit dataset")
    reddit_list = [y for x in os.walk(reddit_dir) for y in glob(os.path.join(x[0], '*.csv'))]
    print(len(reddit_list))
    for i, r in enumerate(reddit_list):
        print(i, r)
        df_reddit = pd.read_csv(r)
        df_reddit['text'] = df_reddit['text'].astype(str)
        if 'comment_wordcount' in df_reddit.columns:
            del df_reddit['comment_wordcount']
        # df_reddit = get_moralstrength(df_reddit)
        # df_reddit = get_nrc_sentiments(df_reddit)
        # df_reddit = get_sentistrength(df_reddit)
        file_type = "comment" if "comment" in r else "post"
        print(file_type)
        df_reddit = get_wordcount(df_reddit, file_type)
        df_reddit.to_csv(r, index=False, index_label=False)