import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt


N = 5
beta = 0.051
beta_step = 0.002
initial_log_depth = 1
mult_factor = 2

save_file = 'output.csv'


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

def get_prob(M, x, y, beta):
    c = count_chi(M, x, y)
    a = np.exp( 2 * beta * c )
    return a / (a+1)

def step(M, x, y, r, beta):
    threshold = get_prob(M, x, y, beta)
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

with open(save_file, 'a') as ff:
    ff.write('----  generated with convol.py, with N={}\n'.format(N))

# plt.axis([0, 0.8, 0, 100])
plt.grid(True)

def get_random_isning(beta, initial_log_depth=1):
    log_depth = initial_log_depth
    initial_depth = 2**initial_log_depth
    depth = initial_depth
    rarr = np.zeros((depth,), dtype=float)
    narr = np.zeros((depth,2), dtype=int)
    reuse_lim = -1
    while True:
        up = np.ones((N,N), dtype=int)
        down = np.full((N,N), -1, dtype=int)
        for n in range(depth):
            if n > reuse_lim:
                rarr[n] = rnd.random()
                narr[n,0] = rnd.randint(0, N)
                narr[n,1] = rnd.randint(0, N)
            r = rarr[n]
            x = narr[n,0]
            y = narr[n,1]
            step(up, x, y, r, beta)
            step(down, x, y, r, beta)
        if check_coalescence(up, down):
            break
        else:
            reuse_lim = depth - 1
            depth = max(depth+1, int(depth*mult_factor))
            log_depth += 1
            rarr.resize(depth)
            narr.resize((depth,2))
    s = np.sum(up)
    frac = (s / N**2 + 1) / 2
    with open(save_file, 'a') as ff:
        ff.write('{};{}\n'.format(beta, frac))
    return up

def observable(mat, beta):
    s = 0
    s2 = 0
    for i in range(N):
        for j in range(N):
            c = count_chi(mat, i, j)
            s += c
            s2 += c * c
    # return (s * s - s2) * beta * beta
    return s

while True:
    mat = get_random_isning(beta)
    en = observable(mat, beta)
    plt.scatter(beta, en, c='#6666ff', marker='x')
    plt.pause(0.001)
    beta += beta_step


