import csv

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


data = load_data()


def tokenize(row):
    def _tokenize_text(text):
        try:
            tokens = komoran.pos(text)
        except Exception as e:
            print(title, e)
            tokens = None
        return tokens

    url, datetime_str, title, content = row
    title_tokens = _tokenize_text(title)
    content_tokens = _tokenize_text(content)
    return url, datetime_str, title, content, title_tokens, content_tokens


def write_tokenized_data(data):
    with open("./data/baseball_tokenized.csv", "w") as fw:
        writer = csv.writer(fw)
        writer.writerow([
            "url", "datetime_str", "title", "content", "title_tokens", "content_tokens"
        ])
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    tokenized_data = process_map(tokenize, data, max_workers=4, chunksize=1)
    write_tokenized_data(tokenized_data)
