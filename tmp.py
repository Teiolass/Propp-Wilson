import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

input_path = 'output2.csv'
N = -1

def reader():
    with open(input_path, 'r') as ff:
        for line in ff:
            if '----' in line:
                N = int(line.split('=')[1])
            elif ';' in line:
                line = line.split(';')
                x = float(line[0])
                y = float(line[1])
                yield (x, y)

theta = 0.3
step = 0.000001

x = 0
y = 0

plt.axis([0, 1, -.05, 1.05])
plt.grid(True)

def is_greater_than_crit(frac):
    return frac < thresholds[0] or frac > thresholds[1]

counter = 0
for (beta, frac) in reader():

    lin = - x * beta + y 
    xp = np.exp(lin)
    pred = 1 / (1 + xp)
    der = xp * pred * pred
    dy = 2 * (pred - frac) * der * step
    dx = dy * beta
    x -= dx
    y -= dy

    theta = -y / x
    print(theta)

    # plt.scatter(counter, theta, c='#6666ff', marker='x')
    # plt.pause(0.00001)

# plt.show()
    
