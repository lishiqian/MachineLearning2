import multiprocessing as mp
import time


def job(val, num, lock):
    lock.acquire()
    for _ in range(10):
        val.value += num
        time.sleep(0.1)
        print(val.value)
    lock.release()


if __name__ == '__main__':
    lock = mp.Lock()
    val = mp.Value('i', 1)
    p1 = mp.Process(target=job, args=(val, 1, lock))
    p2 = mp.Process(target=job, args=(val, 3, lock))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
