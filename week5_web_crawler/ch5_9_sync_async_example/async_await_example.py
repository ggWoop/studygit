import asyncio
import random
import time


async def remote_request():
    print("remote request called!")
    await asyncio.sleep(1)
    return random.randint(0, 10)


async def main():
    start = time.time()
    result_one = await remote_request()
    result_two = await remote_request()
    result_three = await remote_request()
    result_four = await remote_request()
    result_five = await remote_request()

    result = [result_one, result_two, result_three, result_four, result_five]
    print(result)
    print("elapsed: ", time.time() - start)


if __name__ == '__main__':
    asyncio.run(main())
