import glob
import json
import os
PROCESSED_DATA_PATHS = [f'..\\data\\processed_data\\2019\\{sub}' for sub in range(1, 6)]
global_keywords = {}


def create_index(file):
    global global_keywords

    with open(file, 'r') as f:
        content = json.loads(f.read())

    name_parts = file.split(os.sep)
    year = 2019
    month = name_parts[-3]
    day = name_parts[-2]
    file_name = name_parts[-1]

    keywords = content['keywords']
    for kw in keywords:
        kw = kw[0]
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


if __name__ == '__main__':
    files = [file for path in PROCESSED_DATA_PATHS for file in glob.glob(f'{path}/*')]
    files = [file for path in files for file in glob.glob(f'{path}/*')]

    [create_index(file) for file in files]

    with open('../data/global_keywords_index.json', 'w') as f:
        f.write(json.dumps(global_keywords))