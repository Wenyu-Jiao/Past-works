#!/usr/bin/env python
# coding: utf-8

# In[129]:

'''
Note that you need to use the latest version of python3 !!!!!!!!!!
'''    
import gym
import numpy as np
import time
# import math as m
import matplotlib.pyplot as plt
env_name = 'CartPole-v0'
env = gym.make(env_name)

def initialize_states(num_states):
    '''
        the first state is the position of the cart, the second the velocity of the cart
        third the angle of the pole, the forth is the angular velocity of the pole   
    
    Inputs
    ---------------------------------
    num_states: how many bins for each parameter
        type: int
    
    Outputs
    ---------------------------------
    states: list of all bins of the parameters
        type: list
        '''
    states = [
        np.linspace(-4.8, 4.8, num_states),
        np.linspace(-4, 4, num_states),
        np.linspace(-0.418, 0.418,  num_states),
        np.linspace(-4, 4, num_states)]
    return states

def ini_q_table(n):
    '''
    num_states_in_each_v: n
    '''
    qTable = np.random.uniform(low=0,high=1,size=([n] * 4)+[2])    
    return qTable

def digitize_state(state, states):    # return the location index in the qtable
    digi = []
    for i in range(4):
        digi.append(np.digitize(state[i], states[i]) - 1) 
    return tuple(digi)

def do_one_epi(α,qTable):
    γ = 0.95
    reward = 0
    n = 20 ##num_states_in_each_v: n
    states = initialize_states(n)
    state,_ = env.reset()
    digi = digitize_state(state, states)
    # digi_list.append(digi)
    #qTable = initialize_states_and_qtable(n)
    done = False
    x=[]
    θ=[]
    r = []
    howmany_ite_fail = 0
    while not done:
        howmany_ite_fail+=1
        reward += 1
        r.append(reward)
        if np.random.random() > 0.2: #epsilon-greedy policy         
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ##this is expoitation
            action = np.argmax(qTable[digi])
        else:
            ##this is exploration
            action = np.argmin(qTable[digi])
        Q_s_a = qTable[digi + (action, )]     
        new_s, re, done, _, _ = env.step(action)
        x.append(new_s[0])
        θ.append(new_s[2]) 
        #get the location index of the new state   
        newdig = digitize_state(new_s, states)
        maxnewQ = np.max(qTable[newdig])       
        qTable[digi + (action, )] = (1 - α) * Q_s_a + α * (re + γ * maxnewQ)       
        print(reward)
        state = new_s
        digi = digitize_state(state, states)
    else:
        Q_s_a = qTable[digi + (action,)] 
        maxnewQ = 0
        re = reward
        if howmany_ite_fail< 30:
            re += -8000
        elif howmany_ite_fail<60:
            re += -5000
        elif howmany_ite_fail<100:
            re += -3000
        elif howmany_ite_fail<200:
            re += -1080
        elif howmany_ite_fail<300:
            re += -200
        qTable[digi + (action, )] = (1 - α) * Q_s_a + α * (re + γ * maxnewQ)
        return x, θ, r, reward, howmany_ite_fail,qTable,action


# In[130]:


def get_ave_10(rewards):
    '''
    2000 episodes will have 2000 reward, we reshape to 2000/10, 10
    take the average over the row -->get the average reward
    '''
    rewards = np.array(rewards)
    a = np.reshape(rewards,(2000, 10))
    r = np.mean(a, axis=1)
    return r
def more_epi():
    episodes = 20000
    fig, ax = plt.subplots(4, 3, figsize=(15,18))
    α = [0.09, 0.1, 0.11]
    for i in range(len(α)):
        n = 20
        qTable = ini_q_table(n)
        sample_x = []
        sample_theta = []
        cumu_r_list = []
        cumu_r = 0
        rewards = []
        collect_100_not_fail = 0
        for m in range(episodes):
            x, θ, r, reward,howmany_ite_fail,nqTable,action = do_one_epi(α[i],qTable)
            if howmany_ite_fail>195:
                collect_100_not_fail+=1
                sample_x = x
                sample_theta = θ
                if collect_100_not_fail>100:
                    # sample_x = x
                    # sample_theta = θ
                    # plt.figure()
                    # plt.plot(sample_x)
                    # plt.plot(sample_theta)
                    # plt.show()
                    # ax[2,0].plot(x)
                    # ax[2,0].set_title('x sample vs stemstep')
                    # ax[2,1].plot(θ)
                    # ax[2,1].set_title('θ sample vs stemstep')
                    break
            
            else:
                collect_100_not_fail=0
            qTable = nqTable
            #record TD part b
            cumu_r += reward
            cumu_r_list.append(cumu_r)       
            rewards.append(reward)
        r = get_ave_10(rewards)
        if i == 0:
            ax[0,0].plot(r)
            ax[0,0].set_title('rewards vs episode number α=0.09')
            ax[1,0].semilogx(cumu_r_list)
            ax[1,0].set_title("cumulative rewards vs episode number,α=0.09")
            ax[2,0].plot(sample_x)
            ax[2,0].set_title("x vs time step ,α=0.09")
            ax[3,0].plot(sample_theta)
            ax[3,0].set_title(" θ vs time step ,α=0.09")
            
            
            
        if i == 1:
            ax[0,1].plot(r)
            ax[0,1].set_title('rewards vs episode number α=0.1')
            ax[1,1].semilogx(cumu_r_list)
            ax[1,1].set_title("cumulative rewards vs episode number,α=0.1")
            ax[2,1].plot(sample_x)
            ax[2,1].set_title("x vs time step ,α=0.1")
            ax[3,1].plot(sample_theta)
            ax[3,1].set_title(" θ vs time step ,α=0.1")
        if i == 2:
            ax[0,2].plot(r)
            ax[0,2].set_title('rewards vs episode number α=0.11')
            ax[1,2].semilogx(cumu_r_list)
            ax[1,2].set_title("cumulative rewards vs episode number,α=0.11")
            ax[2,2].plot(sample_x)
            ax[2,2].set_title("x vs time step ,α=0.11")
            ax[3,2].plot(sample_theta)
            ax[3,2].set_title(" θ vs time step ,α=0.11")
        # return sample_x, sample_theta    
        #plt.xlabel("episode number(average over 10 trial)")
        #plt.ylabel("average rewards")
        #plt.plot(r)
        

more_epi()        

# animation = True
# if animation is True:
#     sim_name = "CartPole-v0"
#     sim = gym.make(sim_name, render_mode="human")
#     stateinit_Sim, info = sim.reset()
#     stateinit_Sim_index = initialize_states(stateinit_Sim)
#     state_Sim_cur = stateinit_Sim
#     stateIdx_Sim_cur = stateinit_Sim_index
#     qTable = ini_q_table(20)

#     for k in range(1000):
#         x, θ, r, reward,howmany_ite_fail,nqTable,action = do_one_epi(0.1,qTable)
#         state_Sim_nex, reward, terminated, truncated, _ = sim.step(action)
#         stateIdx_Sim_cur = initialize_states(state_Sim_nex)
#         if terminated:
#             print("Animation End! Number of Actions: ", k)
#             break
#         sim.render()


# In[ ]:





