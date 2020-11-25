import matplotlib.pyplot as plt

save_file = 'ssh_out.csv'

plt.axis([0, 0.8, 0, 1])
plt.grid(True)
dic = {}
with open(save_file, 'r') as ff:
    for line in ff:
        # input('continue?')
        plt.close()
        # plt.grid(True)
        if '----' in line:
            # ATTENTION HERE
            print(line[40:-1])
            dic = {}
            continue
            for beta in dic:
                x = dic[beta]
                sm = 0
                cnt = 0
                for mag in x:
                    sm += mag
                    cnt += 1
                if cnt != 0:
                    sm /= cnt
                    plt.scatter(beta, sm, c='#a60d36', marker='.')
            plt.pause(0.0001)
            dic = {}
            print(line)
        else:
            line = line.split(';')
            beta = float(line[0])
            mag = float(line[1])
            # print('{}    {}'.format(beta, mag))
            if beta not in dic:
                dic[beta] = []
            dic[beta].append(mag)
plt.close()
plt.axis([0, 0.8, 0, 1])
plt.grid(True)
for beta in dic:
    x = dic[beta]
    sm = 0
    cnt = 0
    for mag in x:
        sm += mag
        cnt += 1
    if cnt != 0:
        sm /= cnt
        plt.scatter(beta, sm, c='#a60d36', marker='.')
plt.pause(0.0001)
plt.show()


