import numpy as np
from collections import deque

def d_con_lcb(T, K, threshold1, threshold2, gamma):
    """
    Implements the Discounted Constrained LCB algorithm for non-stationary environments.
    """
    env = Environment()

    arm0_discounted_sum = np.zeros(K)
    arm1_discounted_sum = np.zeros(K)
    arm2_discounted_sum = np.zeros(K)
    effective_pulls = np.zeros(K)
    
    for i in range(K):
        sample = env.data_function(i, i, T)
        arm0_discounted_sum[i] = sample[0]
        arm1_discounted_sum[i] = sample[1]
        arm2_discounted_sum[i] = sample[2]
        effective_pulls[i] = 1
            
    a0, a1, a2 = 0.5, 0.5, 0.5

    for t in range(K, T):
        epsilon = 1e-9 

        arm0_discounted_mean = arm0_discounted_sum / (effective_pulls + epsilon)
        arm1_discounted_mean = arm1_discounted_sum / (effective_pulls + epsilon)
        arm2_discounted_mean = arm2_discounted_sum / (effective_pulls + epsilon)
        
        total_effective_pulls = np.sum(effective_pulls)
        exploration_term = np.log(total_effective_pulls)
        
        lcb0 = arm0_discounted_mean - np.sqrt(np.log(2*t**2) / (a0 * (effective_pulls + epsilon)))
        lcb1 = arm1_discounted_mean - np.sqrt(np.log(2*t**2)  / (a1 * (effective_pulls + epsilon)))
        lcb2 = arm2_discounted_mean - np.sqrt(np.log(2*t**2)  / (a2 * (effective_pulls + epsilon)))
        
        plausible_g1 = np.where(lcb1 <= threshold1)[0]
        plausible_g2 = np.where(lcb2 <= threshold2)[0]
        plausible_arr = np.intersect1d(plausible_g1, plausible_g2)
        
        if len(plausible_arr) > 0:
            arm = plausible_arr[np.argmin(lcb0[plausible_arr])] 
        else:
            if len(plausible_g2) > 0:
                arm = plausible_g2[np.argmin(lcb1[plausible_g2])]
            else:
                arm = np.argmin(lcb2)
                
        sample = env.data_function(arm, t, T)
        
        arm0_discounted_sum *= gamma
        arm1_discounted_sum *= gamma
        arm2_discounted_sum *= gamma
        effective_pulls *= gamma
        
        arm0_discounted_sum[arm] += sample[0]
        arm1_discounted_sum[arm] += sample[1]
        arm2_discounted_sum[arm] += sample[2]
        effective_pulls[arm] += 1
        
        env.regret_calculation(arm, threshold1, threshold2)

    feasibility_flag = 1 if len(plausible_arr) > 0 else 0
    #env.plot_regret()

    regret_histories = {
        "suboptimality": env.subopt_regret_history,
        "infeasibility": env.infeas_regret_history
    }
    return (feasibility_flag, regret_histories)
