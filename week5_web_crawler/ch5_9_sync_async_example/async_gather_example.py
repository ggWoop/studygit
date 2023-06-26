import asyncio
import random
import time


async def remote_request():
    print("remote request called!")
    await asyncio.sleep(1)
    return random.randint(0, 10)


async def main():
    start = time.time()
    coroutine_one = remote_request()
    coroutine_two = remote_request()
    coroutine_three = remote_request()
    coroutine_four = remote_request()
    coroutine_five = remote_request()

    result = await asyncio.gather(coroutine_one, coroutine_two, coroutine_three, coroutine_four, coroutine_five)
    print(result)
    print("elapsed: ", time.time() - start)


if __name__ == '__main__':
    asyncio.run(main())
