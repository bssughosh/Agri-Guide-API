import multiprocessing
import time

from humidity_predictions import humidity_caller


def fun1(s, t):
    time.sleep(20)
    print(s)


def fun2(s):
    return s.replace('f', 'e')


if __name__ == "__main__":
    p2 = multiprocessing.Process(target=humidity_caller, args=('maharashtra', 'buldana'))
    p2.start()

    if p2.is_alive():
        p2.join()
    print('Done')
