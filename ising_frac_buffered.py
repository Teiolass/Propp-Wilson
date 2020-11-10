import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

N = 3
beta = 0.35
beta_step = 0.001
initial_log_depth = 1
mult_factor = 2
buffer_size = 30

initial_depth = 2**initial_log_depth
save_file = 'output_bis.csv'

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

def get_prob(M, x, y):
    c = count_chi(M, x, y)
    a = np.exp( 2 * beta * c )
    return a / (a+1)

def step(M, x, y, r):
    threshold = get_prob(M, x, y)
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


plt.axis([0.35, 0.55, -.05, .5])

with open(save_file, 'a') as ff:
    ff.write('----  generated with ising_frac_buffered.py, with N={}\n'.format(N))

while True:
    log_depth = initial_log_depth
    depth = initial_depth
    rarr = np.zeros((depth,), dtype=float)
    reuse_lim = -1
    tot = 0
    for buff in range(buffer_size):
        while True:
            up = np.ones((N,N), dtype=int)
            down = np.full((N,N), -1, dtype=int)
            for n in range(depth):
                if n > reuse_lim:
                    rnd.seed()
                    r = rnd.random()
                    rarr[n] = r
                rnd.seed(int(rarr[n]*34985084))
                x = rnd.randint(0, N)
                y = rnd.randint(0, N)
                r = rnd.random()
                step(up, x, y, r)
                step(down, x, y, r)
            if check_coalescence(up, down):
                break
            else:
                reuse_lim = depth - 1
                depth = max(depth+1, int(depth*mult_factor))
                log_depth += 1
                rarr.resize(depth)
        s = np.sum(up)
        frac = (s / N**2 + 1) / 2
        with open(save_file, 'a') as ff:
            ff.write('{};{}\n'.format(beta, frac))
        tot += (frac - 0.5)**2
    tot /= (buffer_size-1)
    plt.scatter(beta, tot, c='#6666ff', marker='x')
    plt.pause(0.00001)
    beta += beta_step
    if beta > 0.55:
        beta = 0.35

plt.show()
