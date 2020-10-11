import pandas as pd
from glob import glob
import os
import numpy as np

def merge_posts(reddit_dir='../data/rawdata/'):
    sentiment_list = [
        'fear', 'anger', 'anticip', 'trust', 'surprise',
        'positive', 'negative', 'sadness', 'disgust', 'joy',
        'care', 'fairness', 'loyalty', 'authority', 'purity',
        'non_moral', 'sentistrength_pos', 'sentistrength_neg',
        'sentistrength_neutral'
    ]
    post_columns = [
        'name', 'predicted_gender', 'post_wordcount', 'title_wordcount',
        'ups', 'author_premium', 'is_self', 'no_follow', 'num_comments', 'title', 'selftext'
    ] + sentiment_list
    post_list = [y for x in os.walk(reddit_dir) for y in glob(os.path.join(x[0], '*.csv'))]
    post_list = [p for p in post_list if 'comment' not in p]
    post_df = pd.DataFrame()
    for p in post_list:
        print(p)
        df = pd.read_csv(p)
        for c in post_columns:
            if c not in df.columns:
                df[c] = np.nan
        post_df = pd.concat([post_df, df[post_columns]])
    post_df = post_df.groupby("name", as_index=False).apply(lambda x: x.sort_values(by="ups").reset_index().loc[x.shape[0]-1])
    del post_df['index']
    post_df = post_df.drop_duplicates()
    return post_df

def merge_comments(reddit_dir='../data/rawdata/'):
    sentiment_list = [
        'fear', 'anger', 'anticip', 'trust', 'surprise',
        'positive', 'negative', 'sadness', 'disgust', 'joy',
        'care', 'fairness', 'loyalty', 'authority', 'purity',
        'non_moral', 'sentistrength_pos', 'sentistrength_neg',
        'sentistrength_neutral'
    ]
    comment_columns = [
        'id', 'link_id', 'text', 'predicted_gender', 'comment_wordcount', 'subreddit'
    ] + sentiment_list
    comment_list = [y for x in os.walk(reddit_dir) for y in glob(os.path.join(x[0], '*.csv'))]
    comment_list = [p for p in comment_list if 'comment' in p]
    comment_df = pd.DataFrame()
    for p in comment_list:
        print(p)
        df = pd.read_csv(p)
        comment_df = pd.concat([comment_df, df[comment_columns]])
    comment_df = comment_df.groupby("id", as_index=False).apply(lambda x: x.reset_index().loc[x.shape[0]-1])
    del comment_df['index']
    comment_df = comment_df.drop_duplicates()
    return comment_df

def combine_posts_comments(posts, comments):
    df = pd.merge(
        comments, posts, how='inner',left_on='link_id', right_on='name',
        suffixes=["_comment", "_post"]
    )
    df.rename(
        columns={"title": "post_title", "selftext": "post_text", "text": "comment_text"}, inplace=True
    )
    df = df.drop_duplicates()
    return df

if __name__ == "__main__":
    post_df = merge_posts()
    print(post_df.head())
    post_df.to_csv('../data/posts.csv', index=False, index_label=False)
    comment_df = merge_comments()
    print(comment_df.head())
    comment_df.to_csv('../data/comments.csv', index=False, index_label=False)
    df = combine_posts_comments(post_df, comment_df)
    print("writing to csv")
    df.to_csv("../data/reddit_data.csv", index=False, index_label=False)
    # df.loc[df['link_id'] == "t3_e8qkg7"].to_csv("../data/reddit_data_sample.csv", index=False, index_label=False)