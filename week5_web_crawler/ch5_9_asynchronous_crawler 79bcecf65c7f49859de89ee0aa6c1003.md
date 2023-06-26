# ch5_9_asynchronous_crawler

## requests를 이용해서 크롤러를 개발했을 때의 단점

python requests와 for문을 이용해서 크롤러를 개발할 수 있었습니다.

```python
for i in range(num_pages):
	url = f"http://...?page={i}"
	resp = requests.get(url)
	...
```

이 경우, HTTP 요청을 서버로 보내고 응답이 올 때까지 기다렸다가, 응답이 오면 그 다음 요청을 보냅니다. 때문에 서버에서 응답을 내려주는 시간 동안 우리 python 코드는 기다릴 수 밖에 없습니다. 이러한 방식을 동기식(synchronous) 프로그래밍이라고 부릅니다.

간단한 예제를 하나 만들어보겠습니다. 요청을 하나씩 보내고, 각각의 요청마다 1초씩 딜레이가 되어 10번의 요청을 모두 처리하는데 10초가 소요됩니다.

```python
import time
import random

def remote_request():
    time.sleep(1)
    return random.randint(0, 10)

def sync_crawl():
    result = []
    for i in range(10):
        resp = remote_request()
        result.append(resp)
    print(result)

if __name__ == '__main__':
    sync_crawl()
```

이번에는 async 방식을 이용해서 코드를 작성해보겠습니다. 이번에는 10개의 요청을 한꺼번에 보내고, 각각이 1초를 딜레이 한 뒤, 리턴을 해줍니다. 이 경우, 10개의 요청이 모두 처리되는데 약 1초가 걸립니다.

```python
import asyncio
import random

async def remote_request():
    await asyncio.sleep(1)
    return random.randint(0, 10)

async def main():
    result = await asyncio.gather(*[remote_request() for x in range(10)])
    print(result)

if __name__ == '__main__':
    asyncio.run(main())
```

python asyncio의 코드가 다소 난해하여 이해가 가지 않더라도  괜찮습니다. 이번 챕터에서는 sync 방식과 async 방식의 차이만 느끼면 됩니다. 

## Sync와 Async

![Untitled](ch5_9_asynchronous_crawler%2079bcecf65c7f49859de89ee0aa6c1003/Untitled.png)

앞선 예제에서 봤던 차이는 sync(동기) 방식과 async(비동기) 방식의 차이입니다. sync 방식에서는 호출을 넣어놓고 응답이 올 때까지 기다렸다가 다음 동작을 수행합니다. 

반면 비동기 방식에서는 호출을 넣음과 동시에 다음 작업을 수행합니다. 그러다가 호출을 넣어놓은 쪽에서 작업을 완료한 뒤, 응답을 주면은 그제서야 작업을 수행합니다. 위 예제에서는 1초를 기다렸다가 랜덤한 숫자를 리턴해주는 함수를 10번 비동기 방식으로 호출한 뒤, 10개의 함수가 모두 완료되기를 asyncio.gather 함수를 이용해서 기다렸습니다. 때문에 1초가 딜레이 걸리는 함수를 10번 호출하여도 프로그램이 완료되는데 1초가 걸렸던 것입니다.

아직은 python 프로그래밍에 익숙하지 않기 때문에 비동기 프로그래밍 방식이 이해가 가지 않을 수 있습니다. 비동기 프로그래밍에 대한 이론적 설명을 더 하기 보다는 직접 비동기 방식이 적용된 크롤링 프레임워크로 크롤러를 짜보면서 그 차이를 느껴보는 것이 훨씬 좋습니다.

## 웹 크롤링에서의 비동기 프로그래밍

이전에 python requests를 이용하여 크롤러를 개발했을 때에는 요청을 하나 보내고 응답이 올 때까지 기다렸다가 다음 요청을 보내야 했습니다. 때문에 한번에 하나의 HTTP 요청 밖에는 보내지 못했습니다. 하지만 동시에 여러개의 요청을 보내고, 응답이 오면 그 때 파싱하여 결과를 저장하면 훨씬 빠르게 크롤링을 할 수 있지 않을까요?

비동기 방식을 이용하면 이러한 동시에 여러개의 요청을 보내는 방식이 가능해집니다. 그리고 이를 편하게 다룰 수 있도록 도와주는 scrapy라는 프레임워크가 있습니다. 이를 직접 사용해보면서 비동기 프로그래밍의 강력함을 간접 체험해보도록 하겠습니다.

## 정리

이번 챕터에서는 동기와 비동기 프로그래밍에 대해서 간단히 살펴보았습니다. 그리고 웹 크롤링에서 왜 비동기 프로그래밍이 강력한지 짚어보았습니다. 다음 챕터에서는 직접 python 비동기 웹 크롤링 프레임워크인 scrapy를 사용하여 네이버 스포츠 뉴스 크롤러를 개발해보겠습니다.