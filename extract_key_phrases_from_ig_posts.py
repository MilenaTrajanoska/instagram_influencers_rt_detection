import glob
import concurrent
from keybert import KeyBERT
import json
from datetime import datetime
import os
from more_itertools import grouper
import logging
logging.basicConfig(filename='./logs/partitioning.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


POSTS_DATA_PATH = './data/posts_info/info'
PROCESSED_DATA_PATH = './data/processed_data'
NGRAM_LENGTH = 4
kw_model = KeyBERT()
global_keywords = {}
INDEXED_FILES = 0

executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)


def checkpoint_index():
    global INDEXED_FILES
    global global_keywords

    with open(f'./data/global_keywords_index.json', 'r') as f:
        content = json.loads(f.read())
        new_dict = content | global_keywords
    with open(f'./data/global_keywords_index.json', 'w') as f:
        f.write(json.dumps(new_dict))
    global_keywords = {}


def extract_key_phrases_from_group(file_group):
    for file in file_group:
        try:
            extract_key_phrases_from_file(file, NGRAM_LENGTH)
        except (RuntimeError, TypeError, NameError, Exception) as err:
            logging.error(err)


def add_entry_to_global_keywords_index(year, month, day, keywords, file_name):
    global INDEXED_FILES
    global global_keywords

    if INDEXED_FILES >= 1000:
        checkpoint_index()
        INDEXED_FILES = 0

    logging.info(global_keywords)

    for kw in keywords:
        if not global_keywords.get(year):
            global_keywords[year] = {}
            global_keywords[year][month] = {}
            global_keywords[year][month][day] = {}
            global_keywords[year][month][day][kw] = [file_name]
        elif not global_keywords.get(year).get(month):
            global_keywords[year][month] = {}
            global_keywords[year][month][day] = {}
            global_keywords[year][month][day][kw] = [file_name]
        elif not global_keywords.get(year).get(month).get(day):
            global_keywords[year][month][day] = {}
            global_keywords[year][month][day][kw] = [file_name]
        elif not global_keywords.get(year).get(month).get(day).get(kw):
            global_keywords[year][month][day][kw] = [file_name]
        else:
            global_keywords[year][month][day][kw].append(file_name)

    INDEXED_FILES += 1


def create_partition_for_file_with_timestamp(timestamp, file_name, file_content):
    dt_object = datetime.fromtimestamp(timestamp)
    year = dt_object.year
    month = dt_object.month
    day = dt_object.day

    file_partition = f'{PROCESSED_DATA_PATH}/{year}/{month}/{day}'

    if not os.path.isdir(f'{PROCESSED_DATA_PATH}/{year}'):
        os.makedirs(file_partition)
    elif not os.path.isdir(f'{PROCESSED_DATA_PATH}/{year}/{month}'):
        os.makedirs(file_partition)
    elif not os.path.isdir(file_partition):
        os.mkdir(file_partition)

    with open(f'{file_partition}/{file_name}', 'w') as f:
        f.write(json.dumps(file_content))
    add_entry_to_global_keywords_index(year, month, day, file_content['keywords'], file_name)


def extract_key_phrases_from_file(file_path, ngram_length, language='english'):
    file_name = file_path.split(os.sep)[-1]
    with open(file_path, 'r') as f:
        content = json.loads(f.read())
        # accessibility_caption = content.get('edge_media_to_tagged_user')
        # if accessibility_caption:
        #     accessibility_caption = accessibility_caption.get('accessibility_caption', '')
        #     if accessibility_caption:
        #         p = re.compile("Image may contain: (.*)")
        #         result = p.search(accessibility_caption)
        #         accessibility_caption = result.group(1) if result else ''
        # else:
        #     accessibility_caption = ''

        post = content['edge_media_to_caption']['edges'][0].get('node', {})
        text = post.get('text')

        if not text:
            raise Exception('The post did not have any content')
        # text = bytearray(text, 'utf-32').decode('utf-16')  # not english

        logging.info(f'text: {text}')

        timestamp = content['taken_at_timestamp']

        logging.info(f'timestamp: {timestamp}')
        # get caption and tags
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, ngram_length),
            stop_words=language,
            use_mmr=True,
            diversity=0.7
        )
        # if accessibility_caption:
        #     keywords.insert(0, (accessibility_caption, 1.0))

        if len(keywords) == 0:
            raise Exception("No keywords found")

        logging.info(f'keywords: {keywords}')

        content['keywords'] = keywords
        create_partition_for_file_with_timestamp(timestamp, file_name, content)


if __name__ == '__main__':
    futures = [
        executor.submit(extract_key_phrases_from_group, post_meta)
        for post_meta in grouper(5, glob.glob(f'{POSTS_DATA_PATH}/*.info'))
    ]
    concurrent.futures.wait(futures)
    checkpoint_index()
