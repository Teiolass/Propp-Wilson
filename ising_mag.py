import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

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

N = 15
beta = 0.1
beta_step = 0.04
times = 5
save_file = 'mag_out.csv'

with open(save_file, 'a') as ff:
    ff.write('----  generated with ising_mag.py, with N={}\n'.format(N))
plt.axis([0, 0.8, 0, 1])
plt.grid(True)

while beta <= 0.8:
    sm = 0
    beta += beta_step
    for _ in range(times):
        mag = np.abs(np.sum(get_ising(N, beta)) / (N*N))
        sm += mag
        with open(save_file, 'a') as ff:
            ff.write('{};{}\n'.format(beta, mag))
        plt.scatter(beta, mag, c='#03588C', marker='.')
        plt.pause(0.001)
    sm /= times
    plt.scatter(beta, sm, c='#A60D36', marker='o')
    plt.pause(0.001)
