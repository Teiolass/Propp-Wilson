import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

N = 4
beta = 0.35
log_depth = 5

initial_depth = 2**log_depth

def count_chi(M, x, y):
    acc = 0
    if x > 0:
        if y > 0:
            acc += M[x-1,y-1]
        if y < N-1:
            acc += M[x-1,y+1]
    if x < N-1:
        if y > 0:
            acc += M[x+1,y-1]
        if y < N-1:
            acc += M[x+1,y+1]
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


depth = initial_depth
rarr = np.zeros((depth,), dtype=float)
reuse_lim = -1
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
        depth *= 2
        log_depth += 1
        rarr.resize(depth)
s = np.sum(up)
frac = (s / N**2 + 1) / 2

print('Fraction of positives: {}'.format(frac))
print('log depth: {}'.format(log_depth))

