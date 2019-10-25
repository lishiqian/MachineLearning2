import multiprocessing as mp


def job(x):
    return x ** 2


if __name__ == '__main__':
    pool = mp.Pool()
    res = pool.map(job, range(10))
    print(res)
    # apply_async 参数不能传递数组  apply_async只能放入一组参数
    res = pool.apply_async(job, (5,))
    print(res.get())
    # 使用迭代的方法调用apply_async
    res_list = [pool.apply_async(job, (i,)).get() for i in range(10)]
    print(res_list)
