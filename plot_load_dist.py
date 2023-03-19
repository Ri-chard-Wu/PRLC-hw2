
from matplotlib import pyplot as plt

fig, axs = plt.subplots(4)
fig.suptitle('Vertically stacked subplots')


for i in range(4):
    file = open('loadDist_pid' + str(i) + '.txt', 'r')
    Lines = file.readlines()
    a = []
    for line in Lines:
        a.append(int(line.strip()))
        
    axs[i].plot(a)
    # axs[i].set_title('Axis [1, 1]')

# plt.tight_layout()
plt.show()