import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CMS_HLT_env(gym.Env):
    """
    Description:
        CMS HLT system is exemplified by 3 sequences (muon,electron,jet) with 3 steps each: L1 unpacked -> improved resolution -> true pT.
The pT of each object is sampled randomly from an exponentially falling pdf, later to be replaced by typical LHC event. In addition there is a Bit to be set, corresponding to the final decision of the OR of all the trigger paths.

The agent can decide to run a subset of modules making up the full sequence or decide on the overall Bit at any stage.

Observation and states are as follows. There are in total 3*3*3=27 states.

Action

Each Episode terminates:
    When Bit is set to either True or False.
    The agent runs full sequences for all three paths, without having decided.

Reward is given:
    according to whether the Bit is set to the true value and how many modules it ran.

    """

    def __init__(self):
        
        self.seed()
        
        ### sample the pTs from some pdf
        self.mupt = np.random.exponential()*50
        self.elept = np.random.exponential()*50
        self.jetpt = np.random.exponential()*50

        ### make arrays corresponding to the different HLT sequences
        self.mupt_ar = [np.random.normal(
            self.mupt, self.mupt*0.5), np.random.normal(self.mupt, self.mupt*0.15), self.mupt]
        self.elept_ar = [np.random.normal(
            self.elept, self.elept*0.5), np.random.normal(self.elept, self.elept*0.15), self.elept]
        self.jetpt_ar = [np.random.normal(
            self.jetpt, self.jetpt*0.5), np.random.normal(self.jetpt, self.jetpt*0.15), self.jetpt]

        ### bit to set to default value; 2 means unspecified
        self.hltbit_toset = 2

        ### True HLT bit, OR of various cu
        self.hltbit_true = (self.mupt > 40 or self.elept > 50 or self.jetpt > 70 or (
            self.jetpt > 30 and self.elept > 30))

        ### Define the action space and the observation space; there are 8 possible actions, and quite a few observations 81 + the coninues pT
        boundlow = np.array([0, 0, 0])
        boundhigh = np.array([10e4, 10e4, 10e4])
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Tuple((spaces.MultiDiscrete([3, 3, 3]), spaces.Box(
            boundlow, boundhigh, dtype=np.float32), spaces.Discrete(3)))
        self.state = np.array(
            [0, 0, 0, self.mupt_ar[0], self.elept_ar[0], self.jetpt_ar[0], self.hltbit_toset])

        ### Initial reward
        self.reward = 0
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ### Define the actions
    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        ### State Characterization
        i, j, k, mup, ep, jp, bit = self.state

        ### Action Mapping
        if action == 0:
            i, mup = 1, self.mupt_ar[1]
        if action == 1:
            i, mup = 2, self.mupt_ar[2]
        if action == 2:
            j, ep = 1, self.elept_ar[1]
        if action == 3:
            j, ep = 2, self.elept_ar[2]
        if action == 4:
            k, jp = 1, self.jetpt_ar[1]
        if action == 5:
            k, jp = 2, self.jetpt_ar[2]
        if action == 6:
            bit = 1
        if action == 7:
            bit = 0
            
        ### New State
        self.state = i, j, k, mup, ep, jp, bit
        done = (bit != 2)
        done = bool(done)
        #print(done)
        #print(bit)
        #np.delete(self.action_space,action)
        
        ### Give Reward at the end of each game, note: currently for DN agent reward is only used to set threshold what to train on
        if not done:
            self.reward = self.reward#-0.5
            
        if done and (bool(bit) != self.hltbit_true):
            self.reward = self.reward-20-(i+j+k)
        if done and (bool(bit) == self.hltbit_true):
            self.reward = self.reward+20+(i+j+k) #<-- to be able to separate number of modules run

        ### return useful stuff
        #print(self.state)
        return np.array(self.state), self.reward, done, {}

    
    def reset(self):
        self.seed()
        ### Reset function, similar to initialize
        self.mupt = np.random.exponential()*50
        self.elept = np.random.exponential()*50
        self.jetpt = np.random.exponential()*50
        #self.mupt = np.random.exponential()*50
        #self.elept = np.random.exponential()*50
        #self.jetpt = np.random.exponential()*50
        
        self.mupt_ar = [np.random.normal(
            self.mupt, self.mupt*0.5), np.random.normal(self.mupt, self.mupt*0.15), self.mupt]
        self.elept_ar = [np.random.normal(
            self.elept, self.elept*0.5), np.random.normal(self.elept, self.elept*0.15), self.elept]
        self.jetpt_ar = [np.random.normal(
            self.jetpt, self.jetpt*0.5), np.random.normal(self.jetpt, self.jetpt*0.15), self.jetpt]
        
        self.hltbit_toset = 2
        self.hltbit_true = (self.mupt > 40 or self.elept > 50 or self.jetpt > 70 or (
            self.jetpt > 30 and self.elept > 30))

        self.state = np.array(
            [0, 0, 0, self.mupt_ar[0], self.elept_ar[0], self.jetpt_ar[0], self.hltbit_toset])
        
        self.reward=0
        done=False
        return np.array(self.state), self.reward, done, {}


    ### Function to return some other useful Environment properties, handy for analysis
    def gettrue(self):
        
        return np.array([self.hltbit_true, bool(self.mupt>40), bool(self.elept > 50), bool(self.jetpt > 70), bool(self.jetpt > 30 and self.elept > 30), self.mupt, self.elept, self.jetpt])


