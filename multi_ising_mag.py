import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, Pool
from functools import partial

N = 20
beta = 0.1
beta_step = 0.01
times = 5
save_file = 'mag_out_mean.csv'
n_proc = 3


class Random_gen:
    def __init__(self, N):
        self.N = N
        self.depth = 0
        self.rarr = np.zeros((0,), dtype=float)
        self.narr = np.zeros((0,2), dtype=int)

    def advance_depth(self, nd):
        self.rarr.resize((nd,))
        self.narr.resize((nd,2))
        self.rarr[self.depth:] = rnd.random((nd-self.depth,))
        self.narr[self.depth:] = rnd.randint(0, self.N, size=(nd-self.depth,2))
        self.depth = nd

def multiple_step(M, beta, rarr, narr):
    n = rarr.size
    N = M.shape[0]
    for i in range(n):
        r = rarr[i]
        x = narr[i,0]
        y = narr[i,1]
        acc = 0
        if x > 0:
                acc += M[x-1,y]
        if y > 0:
                acc += M[x,y-1]
        if y < N-1:
                acc += M[x,y+1]
        if x < N-1:
                acc += M[x+1,y]
        a = np.exp( 2 * beta * acc )
        threshold = a / (a+1)
        old = M[x, y]
        if r < threshold:
            M[x, y] = 1
        else:
            M[x, y] = -1

def check_coalescence(M1, M2):
    for i in range(N):
        for j in range(N):
            if M1[i,j] != M2[i,j]:
                return False
    return True


def get_ising(N, beta):
    depth = 1
    ran = Random_gen(N)
    ran.advance_depth(depth)
    while True:
        up = np.ones((N,N), dtype=int)
        down = np.full((N,N), -1, dtype=int)
        multiple_step(up, beta, ran.rarr, ran.narr)
        multiple_step(down, beta, ran.rarr, ran.narr)
        if check_coalescence(up, down):
            break
        else:
            depth = max(depth+1, depth*2)
            ran.advance_depth(depth)
    return up

def get_magnetization(N, beta, times):
    print('Getting mag...')
    mag = 0
    ret = []
    for _ in range(times):
        ret.append((beta,np.abs(np.sum(get_ising(N, beta)) / (N*N))))
    return ret

def reader(que):
    with open(save_file, 'a') as ff:
        ff.write('----  generated with ising_mag.py, with N={}\n'.format(N))
    plt.axis([0, 0.8, 0, 1])
    plt.grid(True)
    print('Entering the reader process...')
    while True:
        x = que.get()
        if x == 'DONE':
            break
        sm = 0
        for (beta, mag) in x:
            with open(save_file, 'a') as ff:
                txt = '{};{}\n'.format(beta, mag)
                ff.write(txt)
                print(txt[:-1])
            sm += mag
        sm /= len(x)
        plt.scatter(beta, sm, c='#A60D36', marker='.')
        plt.pause(0.5)

def feeder(queue, val):
    queue.put(val)


if __name__ == '__main__':
    print('Starting...')
    queue = Queue()
    reader_p = Process(target=reader, args=((queue),))
    reader_p.daemon = True
    reader_p.start()    
    pool = Pool(processes=n_proc-1)
    mcallback = partial(feeder, queue) 
    while beta <= 0.8:
        beta += beta_step
        res = pool.apply_async(get_magnetization, args=(N, beta, times),
                callback=mcallback)
    pool.close()
    pool.join()
    queue.put('DONE')
    reader_p.join()
    plt.show()


