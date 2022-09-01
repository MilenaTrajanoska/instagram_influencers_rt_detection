from sentence_transformers import SentenceTransformer, util
import json
import datetime
import torch
from tqdm import tqdm
import concurrent
import threading
import time
import psutil
import os
from utils.datesoputil import create_dates_range


start_time = time.process_time()

QUERY = 'healthy food'


CURRENT_DATE = datetime.datetime(2019, 3, 1)
TIME_FRAME_WEEKS = datetime.timedelta(weeks=6)
DATES = create_dates_range(CURRENT_DATE, TIME_FRAME_WEEKS, reversed=True)

DEVICE = torch.device('cuda')
TOP_K_SIMILARITY = 0.5

INDEX_PATH = '../data/global_keywords_index.json'

lock = threading.Lock()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
futures = []


def calculate_query_to_keyword_similarity(query, keyword):
    return util.pytorch_cos_sim(query, keyword).item()


def calculate_embedding_and_similarity_for_kw(kw):
    global model
    global DEVICE
    global query_embedding
    global keywords_similarity

    lock.acquire()
    if kw in keywords_similarity.keys():
        lock.release()
        return
    lock.release()

    kw_embedding = torch.tensor(model.encode(kw)).to(DEVICE)
    similarity = calculate_query_to_keyword_similarity(query_embedding, kw_embedding)

    if similarity < TOP_K_SIMILARITY:
        return

    lock.acquire()
    keywords_similarity[kw] = similarity
    lock.release()


def extract_keywords_for_date(date):
    global index
    global files_mapping
    global keywords_similarity
    global futures

    fm = index[str(date.year)][str(date.month)][str(date.day)]
    keywords = fm.keys()
    fm = {k: [f'{date.year}/{date.month}/{date.day}/{file}' for file in v] for k, v in fm.items()}

    files_mapping = files_mapping | fm

    futures.extend([
        executor.submit(calculate_embedding_and_similarity_for_kw, kw)
        for kw in tqdm(keywords)
    ])


if __name__ == '__main__':
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(DEVICE)
    query_embedding = torch.tensor(model.encode(QUERY)).to(DEVICE)

    with open(INDEX_PATH, 'r') as f:
        index = json.loads(f.read())

    keywords_similarity = {}
    files_mapping = {}

    [
        extract_keywords_for_date(date)
        for date in tqdm(DATES)
    ]

    concurrent.futures.wait(futures)

    posts = set()
    for kw in tqdm(keywords_similarity.keys()):
        [posts.add(f) for f in files_mapping[kw]]

    with open('../data/healthy_food_posts/posts.json', 'w') as f:
        f.write(json.dumps(list(posts)))

    print("--- execution time in seconds: %.2f ---" % (time.process_time() - start_time))
    process = psutil.Process(os.getpid())
    print(f'--- maximum memory usage: {process.memory_info().peak_wset} ---')

    with open('../data/healthy_food_posts/kw_similarity.json', 'w') as f:
        f.write(json.dumps(keywords_similarity))