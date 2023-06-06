import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces

def make_env_fn(env_key, render_mode=None, frame_stack=1):
    def _f():
        env = gym.make(env_key, render_mode=render_mode)
        if frame_stack > 1:
            env = FrameStack(env, frame_stack)
        return env
    return _f


# Gives a vectorized interface to a single environment
class WrapEnv:
    def __init__(self, env_fn):
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action):
        if action.ndim == 1:
            env_return = self.env.step(action)
        else:
            env_return = self.env.step(action[0])
        if len(env_return) == 4:
            state, reward, done, info = env_return
        else:
            state, reward, done, _, info = env_return
        if isinstance(state, dict):
            state = state['image']
        return np.array([state]), np.array([reward]), np.array([done]), np.array([info])


    def render(self):
        self.env.render()

    def reset(self, seed=0):
        state, *_ = self.env.reset(seed=seed)
        if isinstance(state, tuple):
            ## gym state is tuple type
            return np.array([state[0]])
        elif isinstance(state, dict):
            ## minigrid state is dict type
            return np.array([state['image']])
        else:
            return np.array([state])
        

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space['image'].shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space['image'].dtype)

    def reset(self, seed=None):
        ob = self.env.reset(seed=seed)[0]
        ob = ob['image']
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), {}

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        # ob, reward, done, info = self.env.step(action)
        ob = ob['image']
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=-1)