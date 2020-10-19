import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

N = 5
iterations = 2000

p = 0.7

def transition(state, r):
    k = 1
    if r > p:
        k = -1        
    return np.clip(state + k, 0, N-1)

states = np.arange(N, dtype=int)
depth = 30
coal_times = 0

dist = np.zeros((N,), dtype=int)

for i in range(iterations):
    for n in range(depth):
        r = rnd.random()
        states = transition(states, r)
    coalesce = True
    for j in range(1, N):
        if states[0] != states[j]:
            coalesce = False
            break
    if coalesce:
        dist[states[0]] += 1
        coal_times += 1

print('Fraction of coalescences: {}'.format(coal_times/iterations))

plt.bar(range(N), dist)
plt.show()

