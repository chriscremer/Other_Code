


from os.path import expanduser
home = expanduser("~")

import sys
sys.path.insert(0, home+'/Documents/ALE/Arcade-Learning-Environment')


import random
import pygame
import numpy as np
import cv2 #pip3 install opencv-python

from ale_python_interface import ALEInterface

from DQN_agent import DQN_agent

def main():

    pygame.init()

    ale = ALEInterface()
    ale.setInt(b'random_seed', 123)
    ale.setBool(b'display_screen', True)
    ale.setInt(b'frame_skip', 4)
    # ale.setFloat(b'repeat_action_probability', .7)
    # ale.setBool(b'color_averaging', True)

    game = 'breakout'  #ACKTR tasks#, 'space_invaders', 'seaquest', 'qbert', 'pong', 'beam_rider', 'breakout'
    rom = home+'/Documents/ALE/roms/supported/' + game + '.bin'
    ale.loadROM(str.encode(rom))

    legal_actions = ale.getLegalActionSet()
    rewards, num_episodes = [], 5

    config = []
    agent = DQN_agent(config)

    for episode in range(num_episodes):
        total_reward = 0

        exp_state = []
        exp_action = 0
        exp_reward = 0
        exp_next_state=[]
        while not ale.game_over():

            #Save frame
            frame = ale.getScreenGrayscale()
            frame = cv2.resize(frame, (84, 84))
            exp_next_state.append(frame)
            #Make action
            action = random.choice(legal_actions)
            reward = ale.act(action)
            total_reward += reward
            exp_reward += exp_reward
            #Make experience
            if len(exp_next_state) == 4:
                state_ready = np.reshape(np.stack(exp_next_state), [4*84,84])
                # cv2.imshow('image',state_ready)
                # cv2.waitKey(0)
                exp_action = action
                if len(exp_state) == 0:
                    exp_state = exp_next_state
                else:
                    experience = [exp_state, exp_action, exp_reward, exp_next_state]
                    exp_reward = 0
                    exp_state = exp_next_state
                    exp_next_state = []


        print('Episode %d reward %d.' % (episode, total_reward))
        rewards.append(total_reward)
        ale.reset_game()

    average = sum(rewards)/len(rewards)
    print('Average for %d episodes: %d' % (num_episodes, average))

if __name__ == '__main__':
    main()









