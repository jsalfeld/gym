import gym
import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam


env = gym.make('CMSHLT-v0')
env.reset()
goal_steps = 10

score_requirement = 24

intial_games = 500000

nEpochs=10

def play_a_random_game_first():
    for step_index in range(goal_steps):
#         env.render()
        print(env.action_space)
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

def model_data_preparation():
    print("Playing random games to prepare the training dataset.....")
    training_data = []
    accepted_scores = []
    for game_index in range(intial_games):
        score = 0
        game_memory = []
        previous_observation = []
        observation, reward, done, info=env.reset()
        previous_observation = observation       
        for step_index in range(goal_steps):

            action = env.action_space.sample()#random.randrange(0, 2)
              
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                
            observation, reward, done, info=env.step(action)
            previous_observation = observation
            
            if done:
                break
            
        score=reward
        
        
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                clas=[0]*8
                clas[data[1]]=1
                training_data.append([data[0],clas])
        
        env.reset()

    return training_data


training_data = model_data_preparation()
print(len(training_data))



def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(250, input_dim=input_size, activation='relu'))
    model.add(Dense(128, activation='relu'))
    #model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())

    return model

def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(X[0]), output_size=len(y[0]))    
    model.fit(X, y, epochs=nEpochs)
    
    model.save("model.h5")
    print("Saved model to disk")

    return model

trained_model = train_model(training_data)

scores = []
choices = []
for each_game in range(100):
    #print("STARTING NEW GAME")
    score = 0
    prev_obs = []
    observation, reward, done, info=env.reset()
    prev_obs = observation
    for step_index in range(goal_steps):
        #print("in: {}".format(prev_obs))
        
        if len(prev_obs)==0:
            action = random.randrange(0,6)
        else:            
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
            
            #print(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
            #print(action)
            
        choices.append(action)
        
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        
        #print("out: {}".format(prev_obs))

        if done:
            
            #print(prev_obs)
            break
    
    score=reward
    env.reset()
    scores.append(score)

print(scores)





