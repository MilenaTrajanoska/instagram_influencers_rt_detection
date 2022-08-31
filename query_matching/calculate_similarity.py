from sentence_transformers import SentenceTransformer, util
import json
import datetime
import heapq
import torch
from tqdm import tqdm


QUERY = 'healthy food'
CURRENT_DATE = datetime.datetime(2019, 3, 1)
TIME_FRAME_WEEKS = datetime.timedelta(weeks=8)
START_DATE = CURRENT_DATE - TIME_FRAME_WEEKS
DATES = [START_DATE + datetime.timedelta(days=x) for x in range((CURRENT_DATE-START_DATE).days)]
DEVICE = torch.device('cuda')

INDEX_PATH = '../data/global_keywords_index.json'


def calculate_query_to_keyword_similarity(query, keyword):
    return util.pytorch_cos_sim(query, keyword)


if __name__ == '__main__':
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(DEVICE)
    query_embedding = torch.tensor(model.encode(QUERY)).to(DEVICE)

    with open(INDEX_PATH, 'r') as f:
        index = json.loads(f.read())

    keywords_similarity = {}
    files_mapping = {}

    for date in tqdm(DATES):
        fm = index[str(date.year)][str(date.month)][str(date.day)]
        keywords = fm.keys()
        files_mapping = files_mapping | fm
        for kw in tqdm(keywords):
            if kw not in keywords_similarity.keys():
                kw_embedding = torch.tensor(model.encode(kw)).to(DEVICE)
                similarity = calculate_query_to_keyword_similarity(query_embedding, kw_embedding)
                keywords_similarity[kw] = similarity

    k_top = int(0.05 * len(keywords_similarity))
    keywords_sorted_by_value = heapq.nlargest(k_top, keywords_similarity, key=keywords_similarity.get)
    print(keywords_sorted_by_value)



