import numpy as np
import matplotlib.pyplot as plt

N = 8000
iterations = 500000
reduct = 1

## target distribution
m = 2*N/3
sig_sq = (N/5)**2
pi = np.fromfunction(lambda i: np.exp(-(i-m)**2/sig_sq), (N,))

## Graph
dist = np.zeros((int(N/reduct),), dtype=int)

print('running the chain...')
## Run the chain
state = np.random.randint(0,N)
for i in range(iterations):
    candidate = np.random.randint(0, N)
    r = np.random.random()
    if r < pi[candidate]/pi[state]:
        state = candidate
    dist[int(state/reduct)] += 1

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar(range(np.size(dist)), dist)
ax1.set_title('MCMC data')
ax2.bar(range(N), pi)
ax2.set_title('original dist')
plt.show()

    
