


import json
from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt

fname=home+'/Documents/tmp/baselines_pong2/progress.json'

with open(fname, 'rt') as fh:
    lines = fh.readlines()

rewards =[]
timesteps = []
for i in range(len(lines)):
    header = json.loads(lines[i])

    # print (header['steps'], header['reward (100 epi mean)'])

    rewards.append(header['reward (100 epi mean)'])
    timesteps.append(header['steps'])


plt.plot(timesteps, rewards)
# plt.x_title('Timesteps')
plt.show()

