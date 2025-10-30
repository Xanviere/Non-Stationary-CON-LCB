import numpy as np

def sw_con_lcb(T, K, threshold1,threshold2, w):
    """
    It's the "learner" that tries to figure out which arm is the best by calling data_function and analyzing the results.
    This version uses a single, universal sliding window for all arms combined and pure LCB logic.
    Parameters:
        T: int, 
            time horizon 
        K: int
            number of arms
        threshold1: float
            threshold1 for mean of second dimension
        threshold2: float
            threshold2 for mean of third dimension 
        w: int
            size of the universal sliding window
    Returns:
        out: tuple
            A tuple containing:
            - np.ndarray: array with number of pulls of each arm and feasibility flag.
            - list: A history of cumulative regret at each time step.
    """    
    env = Environment()
    universal_window = deque(maxlen=w)

    arm0_sums = np.zeros(K)
    arm1_sums = np.zeros(K)
    arm2_sums = np.zeros(K)
    num_pulls_in_window = np.zeros(K)
    
    arm0_means = np.zeros(K)
    arm1_means = np.zeros(K)
    arm2_means = np.zeros(K)
    
    for i in range(K):
        sample = env.data_function(i, i, T)
        universal_window.append((i, sample))
        arm0_sums[i] += sample[0]
        arm1_sums[i] += sample[1]
        arm2_sums[i] += sample[2]
        num_pulls_in_window[i] += 1

    for i in range(K):
        if num_pulls_in_window[i] > 0:
            arm0_means[i] = arm0_sums[i] / num_pulls_in_window[i]
            arm1_means[i] = arm1_sums[i] / num_pulls_in_window[i]
            arm2_means[i] = arm2_sums[i] / num_pulls_in_window[i]
            
    a0, a1, a2 = 0.5, 0.5, 0.5 # 1/(2*variance)

    for t in range(K, T):
        lcb0, lcb1, lcb2 = np.zeros(K), np.zeros(K), np.zeros(K)
    
        valid_indices = np.where(num_pulls_in_window > 0)[0]
        invalid_indices = np.where(num_pulls_in_window == 0)[0]

        

        if len(valid_indices) > 0:
            exploration_term = np.log(2*(min(t, w))**2)    #using instead of 2*T**2 for sliding window to adapt
        
            confidence_bound_0 = np.sqrt(exploration_term / (a0 * num_pulls_in_window[valid_indices]))
            confidence_bound_1 = np.sqrt(exploration_term / (a1 * num_pulls_in_window[valid_indices]))
            confidence_bound_2 = np.sqrt(exploration_term / (a2 * num_pulls_in_window[valid_indices]))
            
            lcb0[valid_indices] = arm0_means[valid_indices] - confidence_bound_0
            lcb1[valid_indices] = arm1_means[valid_indices] - confidence_bound_1
            lcb2[valid_indices] = arm2_means[valid_indices] - confidence_bound_2

        if len(invalid_indices) > 0:
            lcb0[invalid_indices] = -np.inf
            lcb1[invalid_indices] = -np.inf
            lcb2[invalid_indices] = -np.inf

        plausible_g1 = np.where(lcb1 <= threshold1)[0]
        plausible_g2 = np.where(lcb2 <= threshold2)[0]
        plausible_arr = np.intersect1d(plausible_g1, plausible_g2)
        
        if len(plausible_arr) > 0: # feasible situation
            arm = plausible_arr[np.argmin(lcb0[plausible_arr])] 
        else: # infeasible situation
            if len(plausible_g2) > 0: # relax g1
                arm = plausible_g2[np.argmin(lcb1[plausible_g2])] # minimize g1
            else: # relax both 
                arm = np.argmin(lcb2) # minimize g2
                    
        sample = env.data_function(arm, t, T)
        
        if len(universal_window) == w:
            evicted_arm, evicted_sample = universal_window[0]
            arm0_sums[evicted_arm] -= evicted_sample[0]
            arm1_sums[evicted_arm] -= evicted_sample[1]
            arm2_sums[evicted_arm] -= evicted_sample[2]
            num_pulls_in_window[evicted_arm] -= 1
        
        universal_window.append((arm, sample))
        arm0_sums[arm] += sample[0]
        arm1_sums[arm] += sample[1]
        arm2_sums[arm] += sample[2]
        num_pulls_in_window[arm] += 1

        for i in range(K):
            if num_pulls_in_window[i] > 0:
                arm0_means[i] = arm0_sums[i] / num_pulls_in_window[i]
                arm1_means[i] = arm1_sums[i] / num_pulls_in_window[i]
                arm2_means[i] = arm2_sums[i] / num_pulls_in_window[i]
        
        env.regret_calculation(arm, threshold1, threshold2)

    feasibility_flag = 1 if len(plausible_arr) > 0 else 0
    #env.plot_regret()

    regret_histories = {
        "suboptimality": env.subopt_regret_history,
        "infeasibility": env.infeas_regret_history
    }
    return (feasibility_flag, regret_histories)
