import numpy as np
from collections import deque


def cusum_con_lcb(T, K, threshold1, threshold2, M, epsilon, h):
    """
    Implements the CON-LCB algorithm with a CUSUM change point detector.
    """
    env = Environment()

    arm0_means = np.zeros(K)
    arm1_means = np.zeros(K)
    arm2_means = np.zeros(K)
    num_pulls = np.zeros(K)

    g0_plus, g0_minus = np.zeros(K), np.zeros(K)
    g1_plus, g1_minus = np.zeros(K), np.zeros(K)
    g2_plus, g2_minus = np.zeros(K), np.zeros(K)
    
    for i in range(K):
        sample = env.data_function(i, i, T)
        arm0_means[i] = sample[0]
        arm1_means[i] = sample[1]
        arm2_means[i] = sample[2]
        num_pulls[i] = 1
            
    a0, a1, a2 = 0.5, 0.5, 0.5

    for t in range(K, T):
        safe_num_pulls = num_pulls + 1e-9
        
        
        lcb0 = arm0_means - np.sqrt(np.log(2*T**2) / (a0 * safe_num_pulls))
        lcb1 = arm1_means - np.sqrt(np.log(2*T**2) / (a1 * safe_num_pulls))
        lcb2 = arm2_means - np.sqrt(np.log(2*T**2) / (a2 * safe_num_pulls))
        
        # Arm Selection Logic
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
        
        # CUSUM change detection 
        if num_pulls[arm] >= M:
            s0_plus = sample[0] - arm0_means[arm] - epsilon
            s0_minus = -sample[0] + arm0_means[arm] - epsilon
            g0_plus[arm] = max(0, g0_plus[arm] + s0_plus)
            g0_minus[arm] = max(0, g0_minus[arm] + s0_minus)

            s1_plus = sample[1] - arm1_means[arm] - epsilon
            s1_minus = -sample[1] + arm1_means[arm] - epsilon
            g1_plus[arm] = max(0, g1_plus[arm] + s1_plus)
            g1_minus[arm] = max(0, g1_minus[arm] + s1_minus)

            s2_plus = sample[2] - arm2_means[arm] - epsilon
            s2_minus = -sample[2] + arm2_means[arm] - epsilon
            g2_plus[arm] = max(0, g2_plus[arm] + s2_plus)
            g2_minus[arm] = max(0, g2_minus[arm] + s2_minus)

            if any([g0_plus[arm] > h, g0_minus[arm] > h, g1_plus[arm] > h, g1_minus[arm] > h, g2_plus[arm] > h, g2_minus[arm] > h]):
                print(f"Change detected at time t={t} on arm {arm}. Resetting all arms.")
                # LOGIC FIX: Reset ALL arms to force re-exploration of the new environment
                num_pulls.fill(1)
                for i in range(K): # Re-initialize all arms with a fresh sample
                    fresh_sample = env.data_function(i, t, T)
                    arm0_means[i] = fresh_sample[0]
                    arm1_means[i] = fresh_sample[1]
                    arm2_means[i] = fresh_sample[2]
                
                # Reset all CUSUM detectors
                g0_plus.fill(0); g0_minus.fill(0)
                g1_plus.fill(0); g1_minus.fill(0)
                g2_plus.fill(0); g2_minus.fill(0)
            else:
                # If no alarm, update normally
                arm0_means[arm] = (arm0_means[arm] * num_pulls[arm] + sample[0]) / (num_pulls[arm] + 1)
                arm1_means[arm] = (arm1_means[arm] * num_pulls[arm] + sample[1]) / (num_pulls[arm] + 1)
                arm2_means[arm] = (arm2_means[arm] * num_pulls[arm] + sample[2]) / (num_pulls[arm] + 1)
                num_pulls[arm] += 1
        else:
            # If not enough samples for CUSUM, update normally
            arm0_means[arm] = (arm0_means[arm] * num_pulls[arm] + sample[0]) / (num_pulls[arm] + 1)
            arm1_means[arm] = (arm1_means[arm] * num_pulls[arm] + sample[1]) / (num_pulls[arm] + 1)
            arm2_means[arm] = (arm2_means[arm] * num_pulls[arm] + sample[2]) / (num_pulls[arm] + 1)
            num_pulls[arm] += 1

        env.regret_calculation(arm, threshold1, threshold2)

    feasibility_flag = 1 if len(plausible_arr) > 0 else 0
    #env.plot_regret()

    regret_histories = {
        "suboptimality": env.subopt_regret_history,
        "infeasibility": env.infeas_regret_history
    }
    return (feasibility_flag, regret_histories)

