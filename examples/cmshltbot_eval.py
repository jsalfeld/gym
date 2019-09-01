import gym
import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
from numpy import loadtxt
from keras.models import load_model
import matplotlib.pyplot as plt

env = gym.make('CMSHLT-v0')
env.reset()
goal_steps = 10
score_requirement = 1
intial_games = 1000000

def play_a_random_game_first():
    for step_index in range(goal_steps):
#         env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("Step {}:".format(step_index))
        print("action: {}".format(action))
        print("observation: {}".format(observation))
        print("reward: {}".format(reward))
        print("done: {}".format(done))
        print("info: {}".format(info))
        if done:
            break
    env.reset()

play_a_random_game_first()


trained_model = load_model('model.h5')
# summarize model.
#trained_model.summary()

scores = []
choices = []
el=[]
je=[]
c=[]
el1=[]
je1=[]
c1=[]
el2=[]
je2=[]
c2=[]

for each_game in range(10000):
    score = 0
    prev_obs = []
    observation, reward, done, info=env.reset()
    prev_obs = observation
    for step_index in range(goal_steps):
        # Uncomment below line if you want to see how our bot is playing the game.
        # env.render()
        
        if len(prev_obs)==0:
            action = random.randrange(0,6)
        else:            
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        #print(prev_obs[-1])
        
        if done:
            print(new_observation[-1])
            if env.gettrue()[1]==False and reward>0: el.append(env.gettrue()[-2])
            if env.gettrue()[1]==False and reward>0: je.append(env.gettrue()[-1])
            if new_observation[-1]==1 and env.gettrue()[1]==False and reward>0: c.append(91)
            if new_observation[-1]==0 and env.gettrue()[1]==False and reward>0: c.append(13)
                
            
            if env.gettrue()[1]==False and reward<0: el1.append(env.gettrue()[-2])
            if env.gettrue()[1]==False and reward<0: je1.append(env.gettrue()[-1])
            if new_observation[-1]==1 and env.gettrue()[1]==False and reward<0: c1.append(91)
            if new_observation[-1]==0 and env.gettrue()[1]==False and reward<0: c1.append(13)

            if env.gettrue()[1]==False: el2.append(env.gettrue()[-2])
            if env.gettrue()[1]==False: je2.append(env.gettrue()[-1])
            if new_observation[-1]==1 and env.gettrue()[1]==False: c2.append(91)
            if new_observation[-1]==0 and env.gettrue()[1]==False: c2.append(13)
            break
        
    score=reward
    env.reset()
    scores.append(score)


print(scores)
print('Average Score:',sum(scores)/len(scores))
u=[x for x in scores if x<1]
print(len(u))

#print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
plt.scatter(el, je,c=c, alpha=0.2)
plt.savefig("correctmatch.png")
plt.scatter(el1, je1,c=c1, alpha=0.2)
plt.savefig("wrongmatch.png")
plt.scatter(el2, je2,c=c2, alpha=0.2)
plt.savefig("anymatch.png")


