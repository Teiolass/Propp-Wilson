import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy.stats import entropy

N = 2
iterations = 1000
initial_depth = 10
samples = 200

def transition(state, r, k):
    candidate = np.clip(state + k, 0, N-1)
    if r < pi[candidate]/pi[state]:
        return candidate
    else:
        return state

coal_times = 0

res = np.zeros((iterations, 2))

for i in range(iterations):
    pi = rnd.rand(N)
    dist = np.zeros((N,), dtype=int)
    for k in range(samples):
        states = np.array([0, N-1])
        depth = initial_depth
        rarr = np.zeros((depth,), dtype=float)
        reuse_lim = -1
        while True:
            for n in range(depth):
                if n > reuse_lim:
                    rnd.seed()
                    r = rnd.random()
                    rarr[n] = r
                rnd.seed(int(rarr[n]*34985084))
                rf = rnd.random()
                k = int(rnd.random()*2)*2 - 1
                func = lambda x: transition(x, rf, k)
                for j in range(2):
                    states[j] = func(states[j])
            if states[0] == states[1]:
                dist[states[0]] += 1
                coal_times += 1
                break
            else:
                reuse_lim = depth - 1
                depth *= 2
                rarr.resize(depth)
    kl = entropy(pi, dist)
    print('[{}/{}]\t\t{}'.format(i+1, iterations, kl))
    res[i] = [N, kl]
    if i+1 % 10 == 0:
        N += 10


plt.scatter(res[:,0], res[:,1])
plt.show()
