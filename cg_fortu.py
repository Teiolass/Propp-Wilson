import numpy as np
import numpy.random as rnd
from collections import deque
import matplotlib.pyplot as plt
from hdtgraph.DynamicGraph import Graph


N = 50
beta = 0
beta_step = 0.05
times = 5


class Random_gen:
    def __init__(self, N):
        self.nmax = N * N * 2
        self.depth = 0
        self.rarr = np.zeros((0,), dtype=float)
        self.narr = np.zeros((0), dtype=int)

    def advance_depth(self, nd):
        self.rarr.resize((nd,), refcheck=False)
        self.narr.resize((nd,), refcheck=False)
        self.rarr[self.depth:] = rnd.random((nd-self.depth,))
        self.narr[self.depth:] = rnd.randint(0, self.nmax, size=(nd-self.depth,))
        self.depth = nd

def are_in_same_cc(V, x1, x2):
    que = deque()
    x = x1
    N = V.shape[0]
    adj = [(1,0,0),(0,1,1),(-1,0,0),(0,-1,1)]
    visited = np.zeros((N,N), dtype=bool)
    visited[x] = True
    while len(que) > 0:
        x = que.popleft()
        if x == x2:
            return True
        visited[x] = True
        (i,j) = x
        if i+1 < N and V[i,j,0] == 1 and visited[i+1, j] == 0:
            que.append((i+1, j))
        if j+1 < N and V[i,j,1] == 1 and visited[i, j+1] == 0:
            que.append((i, j+1))
        if i > 0 and V[i-1,j,0] == 1 and visited[i-1, j] == 0:
            que.append((i-1, j))
        if j > 0 and V[i,j-1,1] == 1 and visited[i, j-1] == 0:
            que.append((i, j-1))
    return False

def multiple_step(p, V, narr, rarr):
    depth = rarr.size
    N = V.shape[0]
    g = Graph()
    for i in range(N):
        for j in range(N):
            if i+1 < N and V[i,j,0] == 1:
                g.insert_edge((i+1, j),(i,j))
            if j+1 < N and V[i,j,1] == 1:
                g.insert_edge((i, j+1),(i,j))
            if i > 0 and V[i-1,j,0] == 1:
                g.insert_edge((i-1, j),(i,j))
            if j > 0 and V[i,j-1,1] == 1:
                g.insert_edge((i, j-1),(i,j))
    for i in range(depth):
        index = depth - i - 1
        u = rarr[index]
        x = narr[index]
        i = x % N
        x = x // N
        j = x % N
        x = x // N
        d = x % 2
        e = (i,j,d)
        th1 = p / (2-p)
        if d == 0:
            y1 = 1
            y2 = 0
        else:
            y1 = 0
            y2 = 1
        x2 = ((i+y1)%N, (j+y2)%N)
        x1 = (i,j)
        V[i,j,d] = 0
        g.delete_edge(x1,x2)
        if u < th1:
            V[i,j,d] = 1
            g.insert_edge(x1,x2)
        elif u < p and g.connected(x1, x2):
            V[i,j,d] = 1
            g.insert_edge(x1,x2)

def get_clustered(N, p):
    depth = 1
    ran = Random_gen(N)
    ran.advance_depth(depth)
    while True:
        up = np.ones((N,N,2), dtype=bool)
        down = np.zeros((N,N,2), dtype=bool)
        multiple_step(p, up, ran.narr, ran.rarr)
        multiple_step(p, down, ran.narr, ran.rarr)
        if np.array_equal(up, down):
            break
        else:
            depth = max(depth+1, depth*2)
            ran.advance_depth(depth)
    return up

def get_ising(N, beta):
    p = (1 - np.exp(-beta*2))
    V = get_clustered(N, p)
    M = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(N):
            if M[i,j] != 0:
                continue
            sigma = rnd.randint(0,2) * 2 - 1
            que = deque()
            que.append((i,j))
            while len(que) > 0:
                i,j = que.popleft()
                M[i,j] = sigma
                if i+1 < N and V[i,j,0] == 1 and M[i+1, j] == 0:
                    que.append((i+1, j))
                if j+1 < N and V[i,j,1] == 1 and M[i, j+1] == 0:
                    que.append((i, j+1))
                if i > 0 and V[i-1,j,0] == 1 and M[i-1, j] == 0:
                    que.append((i-1, j))
                if j > 0 and V[i,j-1,1] == 1 and M[i, j-1] == 0:
                    que.append((i, j-1))
    return M

color_fore = '#A64444'
color_back = '#F2A663'

plt.axis([0, 0.8, 0, 1])
plt.grid(True)

while beta <= 0.8:
    sm = 0
    beta += beta_step
    for _ in range(times):
        mag = np.abs(np.sum(get_ising(N, beta)) / (N*N))
        sm += mag
        plt.scatter(beta, mag, c=color_back, marker='.')
        plt.pause(0.001)
    sm /= times
    plt.scatter(beta, sm, c=color_fore, marker='o')
    plt.pause(0.001)
    
plt.show()
