import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

N = 30
beta = 0.20
beta_step = 0.0
initial_log_depth = 1
mult_factor = 2

save_file = 'output.csv'

initial_depth = 2**initial_log_depth

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

with open(save_file, 'a') as ff:
    ff.write('----  generated with ising_frac.py, with N={}\n'.format(N))

plt.axis([0, 0.8, -.05, 1.05])
plt.grid(True)

# while True:
for _ in range(100):
    log_depth = initial_log_depth
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
            step(up, x, y, r)
            step(down, x, y, r)
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
    plt.scatter(beta, frac, c='#6666ff', marker='x')
    with open(save_file, 'a') as ff:
        ff.write('{};{}\n'.format(beta, frac))
    plt.pause(0.00001)
    beta += beta_step

# plt.show()
