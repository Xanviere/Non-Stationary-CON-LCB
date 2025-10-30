import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class Environment:
    """
    Manages the true state of the multi-armed bandit environment, including the
    non-stationary change and the correct regret calculation.
    """
    def __init__(self):
        self.arm0_means = np.array([0.3, 0.4, 0.6, 0.2, 0.5]) # g0
        self.arm1_means = np.array([0.4, 0.3, 0.1, 0.9, 1.0]) # g1
        self.arm2_means = np.array([0.3, 0.4, 0.8, 0.2, 1.0]) # g2
        self.cov = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]

        self.subopt_regret_history = []
        self.cumulative_subopt_regret = 0
        self.infeas_regret_history = []
        self.cumulative_infeas_regret = 0

    def data_function(self, arm, t, T):
        """
        Returns a sample from the specified arm and updates the environment's
        state if the non-stationary change point has been reached.
        """
        if t > T / 2:
            self.arm0_means = np.array([0.3, 0.4, 0.25, 0.5, 0.1]) # g0
            self.arm1_means = np.array([0.8, 0.3, 0.1, 0.3, 1.0]) # g1
            self.arm2_means = np.array([0.3, 0.7, 0.4, 0.2, 0.2]) # g2
        mean = [self.arm0_means[arm], self.arm1_means[arm], self.arm2_means[arm]]
        sample = np.random.multivariate_normal(mean, self.cov)
        return sample

    def regret_calculation(self, arm, threshold1, threshold2):
        """
        Calculates regret based on the type of error (suboptimality vs. infeasibility).
        """
        is_feasible = (self.arm1_means[arm] <= threshold1) and \
                      (self.arm2_means[arm] <= threshold2)
        
        if is_feasible:
            feasible_arms_indices = np.where((self.arm1_means <= threshold1) & (self.arm2_means <= threshold2))[0]
            if feasible_arms_indices.size > 0:
                optimal_g0 = np.min(self.arm0_means[feasible_arms_indices])
                chosen_g0 = self.arm0_means[arm]
                self.cumulative_subopt_regret += (chosen_g0 - optimal_g0)
        else:
            violation1 = max(0, self.arm1_means[arm] - threshold1)
            violation2 = max(0, self.arm2_means[arm] - threshold2)
            self.cumulative_infeas_regret += (violation1 + violation2)
        
        self.subopt_regret_history.append(self.cumulative_subopt_regret)
        self.infeas_regret_history.append(self.cumulative_infeas_regret)

    def plot_regret(self):
        """Plots both suboptimality and infeasibility regret over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.subopt_regret_history, label='Suboptimality Regret')
        plt.plot(self.infeas_regret_history, label='Infeasibility Regret')
        plt.title("Cumulative Regret Over Time (Discounted CON-LCB)")
        plt.xlabel("Time Step (t)")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def return_regret(self):
        return self.subopt_regret_history, self.infeas_regret_history


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

if __name__ == "__main__":
    
    T = 50000
    K = 5
    THRESHOLD1 = 0.5
    THRESHOLD2 = 0.4
    
    gamma = 0.9999

    print("--- Discounted CON-LCB ---")
    print(f"Time Horizon (T): {T}")
    print(f"Number of Arms (K): {K}")
    print(f"Constraint 1 (g1_mean) <= {THRESHOLD1}")
    print(f"Constraint 2 (g2_mean) <= {THRESHOLD2}")
    print(f"Note: Environment changes at T/2.\nDiscount Factor (gamma) = {gamma}")

    feasibility_flag, regret_histories = d_con_lcb(T, K, THRESHOLD1, THRESHOLD2, gamma)

    print("\n--- Simulation Results ---")
    print(f"{'Instance was determined to be Feasible' if feasibility_flag == 1 else 'Instance was determined to be Infeasible'}")
    
    if regret_histories["suboptimality"]:
        print(f"Final Suboptimality Regret: {regret_histories['suboptimality'][-1]:.2f}")
    if regret_histories["infeasibility"]:
        print(f"Final Infeasibility Regret: {regret_histories['infeasibility'][-1]:.2f}")
