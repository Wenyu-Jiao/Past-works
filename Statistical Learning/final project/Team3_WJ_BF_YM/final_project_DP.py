# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:07:40 2022

@author: MY248
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
M = 1
m = 0.1
g = -9.8
l = 0.5
mu_c = 5e-4
mu_p = 2e-6
detal_t = 0.02
V = np.zeros((6, 3, 3, 3))
gamma = 0.95

all_states = [(i, j, k, m) for i in range(6) for j in range(3) for k in range(3) for m in range(3)]
policy = {state:1 for state in all_states}


'''   
    input:
    x: position of car
    x_dot: the velocity of car
    theta: the angle of carpole
    theta_dot: the velocity of carpole
    action: F = +10 is equivalent to action 1, otherwise F = -10
    
    output:
    the changes of x and changes of theta
'''

def nonlinear_equ(x, x_dot, theta, theta_dot, action):
    if action ==1:
        F = 10
    else:
        F = -10
    
    theta_double_dot = (g * math.sin(theta) + math.cos(theta)*(-F - m * l * theta_dot**2 * math.sin(theta) + mu_c *np.sign(x_dot))/(M+m) - mu_p * theta_dot /(m * l))/(l *(4/3 - (m * math.cos(theta)**2)/(m + M)))
    x_double_dot = (F + m * l *(math.sin(theta)*theta_dot**2 - theta_double_dot *math.cos(theta)) + mu_c *np.sign(x))/(M+m)  
    
    return theta_double_dot, x_double_dot

'''
    input:
    x: position of car
    x_dot: the velocity of car
    theta: the angle of carpole
    theta_dot: the velocity of carpole
    action: F = +10 is equivalent to action 1, otherwise F = -10
    
    output: next all_states
'''
def system_equ (x, x_dot, theta, theta_dot,action):
    theta_double_dot, x_double_dot = nonlinear_equ(x, x_dot, theta, theta_dot, action)
    x_dot = x_dot + detal_t * x_double_dot
    x = x + detal_t * x_dot
    theta_dot = theta_dot + detal_t * theta_double_dot
    theta = theta + detal_t * theta_dot    
    state = (theta, x, theta_dot, x_dot)

    return state

'''
    input all_states with four variables: (theta, x, theta_dot, x_dot)
    output: digitized_state for four variables
    for the theta_dis and x_dis(state for the angle and state for cart position),
    if those two variables out of the bound, in the output state, those two variables will give us none value
    for the x velocity and carpole velocity 
    
'''
def state_discretize(state):
    theta, x, theta_dot, x_dot = state
    qqq= np.multiply([-12, -6, -1, 0, 1, 6, 12], math.pi / 180)
    www = np.array([-2.4, -0.8, 0.8, 2.4]) 
    eee = np.multiply(np.array([-math.inf, -50, 50, math.inf]),math.pi / 180)
    rrr = np.array([-math.inf, -0.5, 0.5, math.inf])
    theta_dis = np.digitize(theta, bins=qqq)
    x_dis = np.digitize(x, bins = www)
    theta_dot_dis = np.digitize(theta_dot, bins=eee)
    x_dot_dis = np.digitize(x_dot, bins=rrr)
    digitized_state = (theta_dis-1, x_dis-1, theta_dot_dis-1, x_dot_dis-1)  

    if theta_dis ==0 or theta_dis == 7 or x_dis == 0 or x_dis == 4:
        return None 
    else:
        return digitized_state 
    
    
'''
    input: old state and action
    new state is given by the system equation by calculating the four variable changes in four system of equations
    digitalize_new_state is the digitalize of the new state
    In the new state digitalizion, if new state has none value, reward is 0
    otherwise 1
    
    output: 
    value function for next state and corresponding reward for going to the new state
    
'''


def step (x, x_dot, theta, theta_dot ,action, V):
    now_state = system_equ (x, x_dot, theta, theta_dot ,action)
    digitalize_new_state = state_discretize(np.array(now_state)) # 
    
    if  digitalize_new_state is None:
        reward = 0.0
        next_Value = 0.0 
    else:
        reward = 1.0
        next_Value = V[digitalize_new_state]

    return next_Value, reward

'''
input: policy and all_states
In the middle, an inverse discretize is performed, which is used for calculation in the step.
Policy evaluation can use the algorithm of synchronous iterative joint dynamic programming: 
starting from any state value function, according to a given strategy, combined with the Bellman expectation equation,
state transition probability and reward, the state value function is updated synchronously and iteratively until it converges, 
and the The final state-value function under the policy.
output: value function

'''

def policy_evaluation(all_states, policy):
    eva= 0
    gamma = 0.95
    number = 10000
    V = np.zeros((6,3,3,3))
    V_new = np.zeros((6, 3, 3, 3))
       
    for i in range(number):
        delta = 0
        eva+=1
        print ('Hold 0n Secnods:', eva*0.02)
        for digitalize_new_state in all_states:
            theta_lable, x_lable, theta_dot_lable, x_dot_lable = digitalize_new_state    
            theta_list = np.multiply([-10, -4, -0.5, 0.5, 4, 10],math.pi / 180)
            x_list = [-1.6, 0, 1.6]
            theta_dot_list = np.multiply([-100, 0, 100],math.pi / 180)
            x_dot_list = [-40, 0, 40]
            theta = theta_list[theta_lable]
            x = x_list[x_lable]
            theta_dot = theta_dot_list[theta_dot_lable]
            x_dot = x_dot_list[x_dot_lable]
            action = policy[digitalize_new_state] # action at this point is the way to know where the force is moving. The input is 1 or other numbers
            Value_function, reward = step(x, x_dot, theta, theta_dot, action, V) 
            value =  (reward + gamma * Value_function)
            v_V = abs(value - V[digitalize_new_state]) # Compare whether it converges, and get the final value function under the modified policy
            delta = max(delta, v_V) # After the previous step and the next step are basically unchanged, there is no reward and then the output
            V_new[digitalize_new_state] = value
        
        V = np.copy(V_new) # 
        # print(reward)
        #V = V_new
        if(delta < 1e-10):
            break
    return V

'''
input: state policy, value function
It is used to judge what action is used for this
Is it a force to the left or a force to the right?
Extended to consider all all_states and all possible behaviors , 
choose the behavior that maximizes q(s,a) in each state. 
That is, consider a greedy strategy
output : new action
'''

def dec_action(digitalize_new_state,policy,V):
        
        theta_lable, x_lable, theta_dot_lable, x_dot_lable = digitalize_new_state
            
        theta_list = np.multiply([-10, -4, -0.5, 0.5, 4, 10],math.pi / 180)
        x_list = [-1.6, 0, 1.6]
        theta_dot_list = np.multiply([-100, 0, 100],math.pi / 180)
        x_dot_list = [-40, 0, 40]
 
        theta = theta_list[theta_lable]
        x = x_list[x_lable]
        theta_dot = theta_dot_list[theta_dot_lable]
        x_dot = x_dot_list[x_dot_lable] 
        # define behavior
        action_old_action =  policy[digitalize_new_state]
        action_new = -policy[digitalize_new_state]
        # print (action_old_action)
        nextV_old_action, reward_old_action = step(x, x_dot, theta, theta_dot, action_old_action, V)
        next_V, reward = step(x, x_dot, theta, theta_dot, action_new, V)
        dec_old_action = reward_old_action + gamma*nextV_old_action    
        dec_action_new = reward + gamma*next_V        
        if dec_action_new > dec_old_action: 
            policy[digitalize_new_state] = action_new 
        else:
            policy[digitalize_new_state] = action_old_action
'''
input : policy value function
When a policy is given, a value function based on the policy can be obtained, 
and a greedy policy can be obtained based on the generated value function. 
According to the new policy, a new value function can be obtained, 
and a new greedy policy can be generated, so that repeated loop iterations will Finally, 
the optimal value function and the optimal policy are obtained. 
The process in which policies are updated and improved in loop iterations is called Policy Iteration. 
The purpose of strategy iteration is to make the strategy converge to the optimum by iteratively calculating the value function.

output: is it converge

'''
def policy_improvement(policy,V):
    policy_stable = True
    policy_prime = policy.copy() 
    for digitalize_new_state in policy:
        old_action = policy[digitalize_new_state] 
        dec_action(digitalize_new_state, policy_prime, V) 
        if (old_action != policy_prime[digitalize_new_state]):
            policy_stable = False       
    return policy_stable, policy_prime
while True:
    
    decide_break, policy_prime = policy_improvement(policy,V)
    if decide_break == False:
        V = policy_evaluation(all_states, policy) #  What is the value of this state
        decide_break, policy_prime = policy_improvement(policy,V) # 
        policy = policy_prime.copy() # give new policy 
        print(decide_break)
    else:
        V = policy_evaluation(all_states, policy) # What is the value of this state
        decide_break, policy_prime = policy_improvement(policy,V) #
        policy = policy_prime.copy() # 
        print(decide_break)
        if decide_break is True: # 
            break # stop while  


'''
image plots
'''

z = 0
for m in range(3): 
    for n in range(3): 
        x_location = np.zeros(162)
        y_location = np.zeros(162)
        value_function_plot = np.zeros(162)
        policy_plot = np.zeros(162)
        x_dot_location = np.zeros(162)
        theta_dot_location = np.zeros(162)
        for i in range(6):
            for j in range(3):
                theta_lable, x_lable, theta_dot_lable, x_dot_lable = [i,j,m,n]
            
                theta_list = np.multiply([-10, -4, -0.5, 0.5, 4, 10],math.pi / 180)
                x_list = [-1.6, 0, 1.6]
                theta_dot_list = np.multiply([-100, 0, 100],math.pi / 180)
                x_dot_list = [-20, 0, 20]
 
                theta = theta_list[theta_lable]
                x = x_list[x_lable]
                theta_dot = theta_dot_list[theta_dot_lable]
                x_dot = x_dot_list[x_dot_lable]
                theta_dot_location[z] = theta_dot
                x_dot_location[z] = x_dot
                x_location[z] = theta * 180/math.pi
                y_location[z] = x
                value_function_plot[z] = V[i,j,m,n]
                policy_plot[z] = policy[(i,j,m,n)]
                z = z + 1
                # print(z)

        style.use('ggplot')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Value Function\n state %d about theta and x'%( m+1) )
                
        ax.set_xlabel('theta')
        ax.set_ylabel('x')
        ax.set_zlabel('Value function')
        ax.scatter3D(x_location, y_location, value_function_plot, c='r')
        ax.legend(loc='best')

        plt.show()


        style.use('ggplot')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Poilcy\n state %d  about theta and x '%( m+1)  )
                
        ax.set_xlabel('theta')
        ax.set_ylabel('x')
        ax.set_zlabel('Poilcy')
        ax.scatter3D(x_location, y_location, policy_plot, c='g')
        ax.legend(loc='best')

        plt.show()
z= 0
m = 0
for i in range(6):   # 为了输出 theta dot， 固定theta
    for j in range(3): 
        x_location = np.zeros(162)
        y_location = np.zeros(162)
        value_function_plot = np.zeros(162)
        policy_plot = np.zeros(162)
        x_dot_location = np.zeros(162)
        theta_dot_location = np.zeros(162)
        for m in range(3):
            for n in range(3):
                theta_lable, x_lable, theta_dot_lable, x_dot_lable = [i,j,m,n]
                # print(i)
                theta_list = np.multiply([-10, -4, -0.5, 0.5, 4, 10],math.pi / 180)
                x_list = [-1.6, 0, 1.6]
                theta_dot_list = np.multiply([-100, 0, 100],math.pi / 180)
                x_dot_list = [-20, 0, 20]
                
                theta = theta_list[theta_lable]
                x = x_list[x_lable]
                theta_dot = theta_dot_list[theta_dot_lable]
                x_dot = x_dot_list[x_dot_lable]
                theta_dot_location[z] = theta_dot
                x_dot_location[z] = x_dot
                x_location[z] = theta * 180/math.pi
                y_location[z] = x
                value_function_plot[z] = V[i,j,m,n]
                policy_plot[z] = policy[(i,j,m,n)]
                z = z + 1
                

        style.use('ggplot')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Value Function\n state %d  about theta dot and x dot'%(j+1)  )
                
        ax.set_xlabel('theta_dot')
        ax.set_ylabel('x_dot')
        ax.set_zlabel('Value function')
        ax.scatter3D(x_dot_location, theta_dot_location, value_function_plot, c='r')
        ax.legend(loc='best')

        plt.show()


        style.use('ggplot')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Poilcy\n state %d about theta dot and x dot'%(j+1) )
                
        ax.set_xlabel('theta_dot')
        ax.set_ylabel('x_dot')
        ax.set_zlabel('Poilcy')
        ax.scatter3D(x_dot_location, theta_dot_location, policy_plot, c='g')
        ax.legend(loc='best')

        plt.show()





#%%






























