import datetime

import pandas as pd
import numpy as np
import torch
import warnings
import json
import concurrent

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import threading

from utils.datesoputil import create_dates_range

warnings.filterwarnings('ignore')
np.random.seed(123)


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
NETWORK_FILES_PATH = '../data/healthy_food_posts/posts.json'
DATA_PATH = '../data/processed_data'
DEVICE = torch.device('cuda')
ALPHA_DECAY = 0.3

lock = threading.Lock()

CURRENT_DATE = datetime.datetime(2019, 3, 1)
TIME_FRAME_WEEKS = datetime.timedelta(weeks=6)
DATES = create_dates_range(CURRENT_DATE, TIME_FRAME_WEEKS, reversed=True)


def decaying_function(day, cumulative_freq=7):
    if day > len(DATES) - cumulative_freq:
        return ALPHA_DECAY ** (day // cumulative_freq)
    return (1 - ALPHA_DECAY) * (ALPHA_DECAY ** (day // cumulative_freq))


MAP_DATE_TO_COEFFICIENT = {
    date: decaying_function(i) for i, date in enumerate(DATES)
}

print(MAP_DATE_TO_COEFFICIENT)

sentiment_mapping = {
    0: -1.2,  # negative
    1: 1,  # neutral
    2: 1.2,  # positive
}

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE)

nodes = []
edges = []


def encode_and_predict_sentiment(text):
    encoded_input = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=500,
        add_special_tokens=True
    ).to(DEVICE)
    output = model(**encoded_input)
    scores = output[0][0]
    return torch.argmax(scores).item()


def read_single_info_file(file_path):
    global nodes
    global edges

    with open(file_path, 'r') as f:
        content = json.loads(f.read())

    parts = file_path.split('/')
    year = eval(parts[-4])
    month = eval(parts[-3])
    day = eval(parts[-2])

    date = datetime.datetime(year, month, day)
    coeff = MAP_DATE_TO_COEFFICIENT[date]

    node_data = {
        'username': content['owner']['username'],
        'is_verified': 1 if content['owner']['is_verified'] else 0,
        'num_likes': content['edge_media_preview_like']['count'] if 'edge_media_preview_like' in content.keys() else 0,
        'num_comments': content['edge_media_to_parent_comment']['count'] if 'edge_media_to_parent_comment' in content.keys() else 0
    }

    user_meta = influencers_meta[influencers_meta['Username'] == node_data['username']]

    if not len(user_meta):
        return

    followers = user_meta['#Followers'].values[0] + 1
    total_posts = user_meta['#Posts'].values[0] + 1

    edge_data = []
    try:
        tagged_users = content['edge_media_to_tagged_user']['edges']
        node_data['total_tagged_users'] = len(tagged_users)
        node_data['num_tagged_verified_users'] = len([v for v in tagged_users if v['node']['user']['is_verified']])
        for tag in tagged_users:
            if tag['node']['user']['username'] == node_data['username']:
                continue

            tag_edges = {
                'user_owner': tag['node']['user']['username'],
                'user_other': node_data['username'],
                'other_user_verified': 10 if node_data['is_verified'] else 1,
                'owner_verified': 10 if tag['node']['user']['is_verified'] else 1,
                'post_likes': node_data['num_likes'],
                'sentiment_score': sentiment_mapping[2],
                'spam_score': 1,
            }
            tag_edges['weight'] = coeff * tag_edges['sentiment_score'] * tag_edges['owner_verified'] * \
                                  (
                                          (tag_edges['post_likes'] * 10 + node_data['num_comments'] * 100)
                                          / (total_posts * followers)
                                          + node_data['num_tagged_verified_users'] / total_posts
                                   )

            edge_data.append(tag_edges)
    except:
        node_data['total_tagged_users'] = 0
        node_data['num_tagged_verified_users'] = 0

    comments = content['edge_media_to_parent_comment']['edges'] if 'edge_media_to_parent_comment' in content.keys() else []

    for comment in comments:
        try:
            if comment['node']['owner']['username'] == node_data['username']:
                continue

            comment_edges = {
                'user_owner': node_data['username'],
                'user_other': comment['node']['owner']['username'],
                'owner_verified': 10 if node_data['is_verified'] else 1,
                'other_user_verified': 10 if comment['node']['owner']['is_verified'] else 1,
                'post_likes': comment['node']['edge_liked_by']['count'] if 'edge_liked_by' in comment['node'].keys() else 0,
                'spam_score': -1 if comment['node']['did_report_as_spam'] else 1
            }

            comment_text = comment['node']['text']
            comment_edges['sentiment_score'] = sentiment_mapping[
                encode_and_predict_sentiment(comment_text)
            ]
            edge_data.append(comment_edges)
        except:
            continue

    nodes.append(node_data)
    edges.extend(edge_data)


if __name__ == '__main__':
    with open(NETWORK_FILES_PATH, 'r') as f:
        files = json.loads(f.read())

    print(f'Total posts: {len(files)}')

    influencers_meta = pd.read_csv('../data/influencers.txt', sep='\t', header=0)
    influencers_meta.dropna(inplace=True)

    for file in tqdm(files):
        read_single_info_file(f'{DATA_PATH}/{file}')

    nodes = pd.DataFrame(nodes)
    edges = pd.DataFrame(edges)

    nodes = nodes.groupby('username').agg(
        is_verified=pd.NamedAgg(column="is_verified", aggfunc="max"),
        num_likes=pd.NamedAgg(column="num_likes", aggfunc="sum"),
        num_comments=pd.NamedAgg(column="num_comments", aggfunc="sum"),
        total_tagged_users=pd.NamedAgg(column="total_tagged_users", aggfunc="sum"),
        num_tagged_verified_users=pd.NamedAgg(column="num_tagged_verified_users", aggfunc="sum"),
    )

    edges = edges.groupby(['user_owner', 'user_other']).agg(
        owner_verified=pd.NamedAgg(column="owner_verified", aggfunc="max"),
        other_user_verified=pd.NamedAgg(column="other_user_verified", aggfunc="max"),
        post_likes=pd.NamedAgg(column="post_likes", aggfunc="sum"),
        sentiment_score=pd.NamedAgg(column="sentiment_score", aggfunc="sum"),
        is_spam=pd.NamedAgg(column="spam_score", aggfunc="sum"),
        weight=pd.NamedAgg(column='weight',  aggfunc="sum"),
    )

    nodes.to_csv('../data/healthy_food_posts/node_data.csv')
    edges.to_csv('../data/healthy_food_posts/edge_data.csv')
