import os

import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

try:
    import pybullet_envs
except ImportError:
    pass

import numpy as np
from collections import deque
from gym import spaces



# def make_vec_envs(env_id, seed, rank, log_dir):

#     return envs1, envs2



def make_env(env_id, seed, rank, log_dir):
    def _thunk():

        env = gym.make(env_id) #this prints
        # print('here')

        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)

        #so this overwrites the other env? so ill change it
        if is_atari:
            # env = make_atari(env_id)
            #took this from make_atari
            assert 'NoFrameskip' in env.spec.id
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)

        env.seed(seed + rank)

        if log_dir != '':
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        if is_atari:
            env = wrap_deepmind(env)
            env = WrapPyTorch(env)

        return env

    # print (_thunk())
    # fadfa

    return _thunk


def make_env_monitor(env_name, save_dir):
    env = gym.make(env_name) #this prints

    is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)

    if is_atari:
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)

    if is_atari:
        env = wrap_deepmind(env)
        env = WrapPyTorch(env)

    env = gym.wrappers.Monitor(env, save_dir+'/videos/', video_callable=lambda x: True, force=True)
    return env




def make_both_env_types(env_name):
    env = gym.make(env_name) #this prints

    is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)

    if is_atari:
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        # env.seed(seed + rank)

        env2 = wrap_deepmind(env)
        env2 = WrapPyTorch(env2)

    return env, env2




class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)



class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        if isinstance(env.action_space, gym.spaces.MultiBinary):
            self.noop_action = np.zeros(self.env.action_space.n, dtype=np.int64)
        else:
            # used for atari environments
            self.noop_action = 0
            assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs





class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype='uint8')
        self._skip       = skip

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info
