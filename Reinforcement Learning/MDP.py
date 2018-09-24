## Program to evaluate policy
## Source Link : https://towardsdatascience.com/reinforcement-learning-demystified-solving-mdps-with-dynamic-programming-b52c8093c919

import numpy as np
import sys
import grid_world

env = grid_world.GridworldEnv()

def policy_eval(policy, env, discount_factor = 1.0, epsilon = 0.00001):
    
    V_old = np.zeros(env.nS)
    
    while True:
        V_new = np.zeros(env.nS)
        delta = 0
        
        for s in range(env.nS):
            v_fn = 0
            action_probs = policy[s]
            
            for a in range(env.nA):
                [(prob, next_state, reward, done)] = env.P[s][a]
                
                v_fn += action_probs[a] * (reward + discount_factor * V_old[next_state])
                
            delta = max(delta, abs(v_fn - V_old[s]))
            V_new[s] = v_fn
            
        V_old = V_new
        
        if(epsilon > delta):
            break
    return(np.array(V_old))

random_policy = np.ones([env.nS, env.nA])/env.nA

v = policy_eval(random_policy, env)

print(v)
