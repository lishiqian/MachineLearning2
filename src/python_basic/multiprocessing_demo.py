import multiprocessing as mp


def job():
    res = 0
    for i in range(100):
        res += i ** 2
    print(res)


if __name__ == '__main__':
    pross = mp.Process(target=job)
    pross.start()
    pross.join()