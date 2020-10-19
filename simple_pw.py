import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

N = 26
iterations = 50000
initial_depth = 0

## target distribution
m = 2*N/3
sig_sq = (N/3)**2
pi = np.fromfunction(lambda i: np.exp(-(i-m)**2/sig_sq), (N,))
# pi = np.ones((N,))


def transition(state, r):
    rnd.seed(int(r*21349813))
    r = rnd.random()
    candidate = np.random.randint(0, N)
    if r < pi[candidate]/pi[state]:
        return candidate
    else:
        return state

coal_times = 0

dist = np.zeros((N,), dtype=int)

for i in range(20):
    states = np.arange(N, dtype=int)
    while True:
        r = rnd.random()
        func = lambda x: transition(x, r)
        vfunc = np.vectorize(func)
        states = vfunc(states)
        coalesce = True
        for j in range(1, N):
            if states[0] != states[j]:
                coalesce = False
                break
        initial_depth += 1
        if coalesce:
            break
initial_depth = int(initial_depth/20)


for i in range(iterations):
    states = np.arange(N, dtype=int)
    depth = initial_depth
    rarr = np.zeros((depth,), dtype=float)
    reuse_lim = -1
    while True:
        for n in range(depth):
            if n > reuse_lim:
                r = rnd.random()
                rarr[n] = r
            func = lambda x: transition(x, rarr[n])
            vfunc = np.vectorize(func)
            states = vfunc(states)
        coalesce = True
        for j in range(1, N):
            if states[0] != states[j]:
                coalesce = False
                break
        if coalesce:
            dist[states[0]] += 1
            coal_times += 1
            break
        else:
            reuse_lim = depth - 1
            depth *= 2
            rarr.resize(depth)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar(range(np.size(dist)), dist)
ax1.set_title('PW data')
ax2.bar(range(N), pi)
ax2.set_title('original dist')
plt.show()
