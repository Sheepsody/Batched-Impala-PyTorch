"""
Environments and wrappers for MarioKart training.
"""

import gym
import numpy as np

import retro
import cv2

from gym.wrappers import FrameStack


def make_env(state, stacks, size, game='SuperMarioKart-Snes', record=False):
    env = retro.make(
        game=game,
        use_restricted_actions=retro.Actions.ALL,
        state=state
    )
    if record:
        env = KartRecorder(env=env, size=size)
    env = KartMultiDiscretizer(env)
    env = KartObservation(env, size=size)
    env = FrameStack(env, num_stack=stacks)
    # Careful, this has to be done after the stack
    env = KartSkipper(env, skip=5)
    # Has to be done after skipper
    env = KartReward(env)
    return env


class KartRecorder(gym.ObservationWrapper):
    "Recording with OpenCV"
    def __init__(self, env, size, name="record"):
        super(KartRecorder, self).__init__(env)

        # Initializing the movie
        self.video_args = {
            "filename": name+'.avi', 
            "fourcc": cv2.VideoWriter_fourcc('M','J','P','G'), 
            "fps":  60.0, # SNES refresh rate and speed increase
            "frameSize": (256, 224)
        }

    def reset(self, **kwargs):
        observation = super(KartRecorder, self).reset(**kwargs)
        self.video = cv2.VideoWriter(**self.video_args)
        return observation
    
    def observation(self, obs):
        # Issue with the first frame while calling reset 
        if hasattr(self, 'video'):
            if self.video.isOpened():
                self.video.write(obs[..., ::-1])
        return obs
    
    def close(self):
        """Flush all monitor data to disk and close any open rending windows."""
        super(KartRecorder, self).close()

        if hasattr(self, 'video'):
            self.video.release()


class KartObservation(gym.ObservationWrapper):
    """
    Prior operations done on the input images for the neural network
    """
    def __init__(self, env, size=(128, 56)):

        super(KartObservation, self).__init__(env)

        self.size = size
    
    def observation(self, obs):
        # To gray scale
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # Adding a channel to the image
        obs = np.expand_dims(obs, axis=-1)
        # We remove the minimap
        obs = obs[3:106, 0:256, :]
        # Resizing the image
        obs = cv2.resize(obs, self.size)
        # Converting the image to an array
        obs = obs/255.

        return obs


class KartMultiDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it a tuple of mutliple discrete actions for Mario Kart
    The three sub actions are
        1. Controle pad (Right/Left)
        2. Accelerate (B) / Brake (Y)
        3. Hop (L)
        Actions we're not yet considering
        4. Use item (A)
        5. Change view (X)
    """
    # Discretized action space
    discretized_actions = [
        [], # No action is made
        # ["A"] if objects allowed
        ["LEFT"],
        ["RIGHT"],
        ["LEFT", "B"],
        ["RIGHT", "B"],
        ["LEFT", "Y"],
        ["RIGHT", "Y"],
        ["LEFT", "B", "L"], # Power slides
        ["RIGHT", "B", "L"]
    ]

    def __init__(self, env):
        super(KartMultiDiscretizer, self).__init__(env)

        # The map of buttons on snes
        buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]

        # List of equivalents in the Binary format
        self._actions = []
        self.nb_actions = len(KartMultiDiscretizer.discretized_actions)

        for action in KartMultiDiscretizer.discretized_actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        
    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class KartReward(gym.RewardWrapper):
    """
    Performs different kinds of operations on the reward, including asymetric scaling as proposed in IMPALA
    """
    def __init__(self, env, clip_min=-1.0, clip_max=1.0, scale=1.0, asymetric=False):
        super(KartReward, self).__init__(env)

        assert clip_min <= clip_max, "Min and max clipping are not consistent"
        # Operations are presented in the order they should be done
        self._scale = lambda x : x*scale
        self._asym = lambda x : 0.3*min(np.tanh(x), 0.0) + 0.5*max(np.tanh(x), 0.0)
        self._clip = lambda x : np.clip(a_min=clip_min, a=x, a_max=clip_max)

    def reward(self, reward):
        reward = self._scale(reward)
        reward = self._asym(reward)
        reward = self._clip(reward)
        return reward


class KartSkipper(gym.Wrapper):
    """
    Returns only one observations every [skip]-steps
    """
    def __init__(self, env, skip=5):
        super(KartSkipper, self).__init__(env)

        self.skip = skip

    def step(self, action):
        # Check if it's done, accumulate the reward !
        done = None
        acc_reward = 0
        acc_infos = dict()
        # Skipping !
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            acc_reward += reward
            acc_infos.update(info)
            if done :
                break
        return obs, acc_reward, done, acc_infos



def KartDiscreteSkip(KartMultiDiscretizer):
    """
    Each time a new episode is started, the first 
    """
    def __init__(self, env, max_skip):
        super(KartDiscreteSkip, self).__init__(env)

        self.max_skip = max_skip

    def reset(self, **kwargs):
        observation = super(KartDiscreteSkip, self).reset(**kwargs)
        observation, _, _, _ = self.env.step(self._actions[0].copy())
        return observation



# TODO : class to record the episode
