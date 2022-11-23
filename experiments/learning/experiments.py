import os

os.system('python singleagent.py --act vel --algo td3')
os.system('python singleagent.py --act vel --algo sac')


#1000000 4 hrs  save-updown-ddpg-kin-pid-11.20.2022_19.51.45
# return -1 * np.linalg.norm(np.array([1, 1, 1]) - state[0:3]) ** 2 \
#        - np.linalg.norm(np.pi - state[7]) ** 2

#1500000 6 hrs save-updown-ddpg-kin-pid-11.21.2022_02.17.50
# norm_ep_time = (self.step_counter / self.SIM_FREQ) / self.EPISODE_LEN_SEC
# if norm_ep_time > 0.5:
#     return -1 * np.linalg.norm(np.array([1, 1, 1]) - state[0:3]) ** 2
# else:
#     return -1 * np.linalg.norm(np.array([0, 0, 0]) - state[0:3]) ** 2

#### done 500000 best so far
## save-updown-ddpg-kin-pid-11.21.2022_12.32.19