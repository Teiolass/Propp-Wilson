import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pickle
import time

class Random_gen:
    def __init__(self, N):
        self.active_size = 1024 * 1024 * 10
        self.depth = 1
        self.save_path = '/tmp/random_gen_{}.pkl'
        self.filled_to = 0
        self.rarr = np.zeros((self.depth,), dtype=float)
        self.narr = np.zeros((self.depth, 2), dtype=int)
        self.active_chunk = 0
        self.to_emit = 0
        self.N = N

    def get_new_random(self):
        if self.to_emit < self.active_size:
            if self.to_emit < self.filled_to:
                x = self.narr[self.to_emit, 0]
                y = self.narr[self.to_emit, 1]
                r = self.rarr[self.to_emit]
                self.to_emit += 1
                return (x, y, r)
            else:
                if self.to_emit >= self.depth:
                    self.depth = max(self.depth+1, self.depth*2)
                    self.rarr.resize(self.depth)
                    self.narr.resize((self.depth, 2))
                x = rnd.randint(0, self.N)
                y = rnd.randint(0, self.N)
                r = rnd.random()
                self.narr[self.to_emit,0] = x
                self.narr[self.to_emit,1] = y
                self.rarr[self.to_emit] = r
                self.filled_to += 1
                self.to_emit += 1
                return (x, y, r)
        raise IndexError

    def reset_queue(self):
        self.filled_to = 0
        self.rarr = np.zeros((self.active_size,), dtype=float)
        self.narr = np.zeros((self.active_size, 2), dtype=int)
        self.active_chunk = 0
        self.to_emit = 0
        self.depth = 1
        # free the disk space
        
    def back_to_beginning(self):
        # load chunk 0
        self.to_emit = 0

    def back_by(n):
        self.to_emit -= n


def count_chi(M, x, y):
    acc = 0
    if x > 0:
            acc += M[x-1,y]
    if y > 0:
            acc += M[x,y-1]
    if y < N-1:
            acc += M[x,y+1]
    if x < N-1:
            acc += M[x+1,y]
    return acc

def get_prob(chi, beta):
    a = np.exp( 2 * beta * chi )
    return a / (a+1)

def multiple_step(M, chi, prob, beta, ran, n):
    for _ in range(n):
        (x, y, r) = ran.get_new_random()
        threshold = prob[x, y]
        old = M[x, y]
        if r < threshold:
            M[x, y] = 1
        else:
            M[x, y] = -1
        if M[x, y] != old:
            adj = [[-1, 0], [1, 0], [0, 1], [0, -1]]
            nc = 0
            for v in adj:
                x2 = x + v[0]
                y2 = y + v[1]
                if x2 < 0 or x2 >= N:
                    continue
                if y2 < 0 or y2 >= N:
                    continue
                c = M[x2, y2] * old
                chi[x2, y2] -= 2 * c
                nc -= c
                a = np.exp( 2 * beta * c )
                prob[x2, y2] = a / (a+1)
            chi[x, y] = nc
            a = np.exp( 2 * beta * c )
            prob[x, y] = a / (a+1)

def check_coalescence(M1, M2):
    for i in range(N):
        for j in range(N):
            if M1[i,j] != M2[i,j]:
                return False
    return True

def fill_chi(M):
    chi = np.zeros(M.shape, dtype=int)
    N = M.shape[0]
    for i in range(N):
        for j in range(N):
            chi[i, j] = count_chi(M, i, j)
    return chi


def fill_prob(chi, beta):
    prob = np.zeros(chi.shape, dtype=float)
    N = chi.shape[0]
    for i in range(N):
        for j in range(N):
            prob[i, j] = get_prob(chi[i, j], beta)
    return prob

def get_ising(N, beta):
    depth = 1
    ran = Random_gen(N)
    while True:
        up = np.ones((N,N), dtype=int)
        down = np.full((N,N), -1, dtype=int)
        chi_up = fill_chi(up)
        chi_down = fill_chi(down)
        prob_up = fill_prob(chi_up, beta)
        prob_down = fill_prob(chi_down, beta)
        multiple_step(up, chi_up, prob_up, beta, ran, depth)
        ran.back_to_beginning()
        multiple_step(down, chi_down, prob_down, beta, ran, depth)
        if check_coalescence(up, down):
            break
        else:
            depth = max(depth+1, depth*2)
            ran.back_to_beginning()
    return up

N = 25
beta = 0.2
beta_step = 0.00
times = 10
save_file = 'mag_out.csv'

with open(save_file, 'a') as ff:
    ff.write('----  generated with ising_frac.py, with N={}\n'.format(N))
plt.axis([0, 0.8, 0, 1])
plt.grid(True)

# while beta < 0.8:
start_time = time.time()
for _ in range(20):
    beta += beta_step
    mag = np.abs(np.sum(get_ising(N, beta)) / (N*N))
    with open(save_file, 'a') as ff:
        ff.write('{};{}\n'.format(beta, mag))
    plt.scatter(beta, mag, c='#03588C', marker='x')
    # l'altro colore e` #A60D36
    plt.pause(0.001)
    
print("--- %s seconds ---" % (time.time() - start_time))
