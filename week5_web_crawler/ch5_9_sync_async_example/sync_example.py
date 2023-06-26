import time
import random


def remote_request():
    print("remote request called!")
    time.sleep(1)
    return random.randint(0, 10)


def sync_crawl():
    start = time.time()
    result = []
    for i in range(5):
        resp = remote_request()
        result.append(resp)
    print(result)
    print("elapsed: ", time.time() - start)


if __name__ == '__main__':
    sync_crawl()
