import threading as th


def job1():
    global A, lock
    lock.acquire()
    for i in range(20):
        A += 1
        print("job1", A)
    lock.release()


def job2():
    global A, lock
    lock.acquire()
    for i in range(20):
        A += 10
        print("job2", A)
    lock.release()


if __name__ == '__main__':
    lock = th.Lock()
    A = 0
    t1 = th.Thread(target=job1)
    t2 = th.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
