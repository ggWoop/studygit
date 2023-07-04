import csv

from konlpy.tag import Komoran
from tqdm.contrib.concurrent import process_map

komoran = Komoran(userdic="./data/user.dic")

def load_data():
    data = []
    with open('./data/baseball_preprocessed.csv',encoding='utf-8-sig') as fr:
        reader = csv.reader(fr)
        next(reader)
        for row in reader:
            data.append(row)

    return data

data = load_data()


def tokenize(row):
    def _tokenize(text):
        try:
            tokens = komoran.pos(text)
            return tokens
        except Exception as e:
            print(title, e)
            return None

    url, datatime_str, title, content = row

    title_tokens = _tokenize(title)
    content_tokens = _tokenize(content)

    return  url, datatime_str, title, content,title_tokens,content_tokens

def wirte_tokenized_data(data):
    with open('./data/baseball_tokenized.csv','w',encoding='utf-8-sig') as fw:
        writer = csv.writer(fw)
        writer.writerow(['url', 'datatime_str','title' , 'content','title_tokens','content_tokens'])
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    data = load_data()
    tokenized_data = process_map(tokenize, data, max_workers=6, chunksize=1)
    wirte_tokenized_data(tokenized_data)





