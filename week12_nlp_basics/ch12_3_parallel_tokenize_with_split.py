import csv
import os

from konlpy.tag import Komoran
from tqdm.contrib.concurrent import process_map

komoran = Komoran(userdic="./data/user.dic")


def load_data():
    data = []
    with open("./data/baseball_preprocessed.csv") as fr:
        reader = csv.reader(fr)
        next(reader)
        for row in reader:
            data.append(row)
    return data


def tokenize(split_chunk):
    def _tokenize(x):
        try:
            tokens = komoran.pos(x)
            return tokens
        except Exception as e:
            print(e, x)
            return None
    with open(f"./data/tmp_{os.getpid()}.csv", "w") as fw:
        for row in split_chunk:
            url, datetime_str, title, content = row
            writer = csv.writer(fw)
            title_tokens = _tokenize(title)
            content_tokens = _tokenize(content)
            writer.writerow([url, datetime_str, title, content, title_tokens, content_tokens])


def write_tokenized_data(data):
    with open("./data/baseball_tokenized.csv", "w") as fw:
        writer = csv.writer(fw)
        writer.writerow(["url", "datetime_str", "title", "content", "title_tokens", "content_tokens"])
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    data = load_data()
    n_workers = 4
    chunksize = len(data) // n_workers
    split_1 = data[:chunksize]
    split_2 = data[chunksize:chunksize*2]
    split_3 = data[chunksize*2:chunksize*3]
    split_4 = data[chunksize*3:chunksize*4]
    split_list = [split_1, split_2, split_3, split_4]
    tokenized_data = process_map(tokenize, split_list, max_workers=4)
    write_tokenized_data(tokenized_data)
