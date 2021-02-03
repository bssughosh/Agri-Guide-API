import multiprocessing
import time


def fun1(s):
    time.sleep(20)
    print(s)


def fun2(s):
    return s.replace('f', 'e')


if __name__ == "__main__":
    p1 = multiprocessing.Process(target=fun1, args=('ace',), )
    p2 = multiprocessing.Process(target=fun2, args=('face',), )

    p1.start()
    p2.start()

    if p1.is_alive():
        p1.join()
    p2.join()
    print('Done')
