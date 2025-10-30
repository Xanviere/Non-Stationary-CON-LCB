import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class Environment:
    """
    Manages the true state of the multi-armed bandit environment, including the
    non-stationary change and the correct regret calculation.
    """
    def __init__(self):
        # Initial parameters for the first period
        self.arm0_means = np.array([0.3, 0.4, 0.6, 0.2, 0.5]) # g0
        self.arm1_means = np.array([0.4, 0.3, 0.1, 0.9, 1.0]) # g1
        self.arm2_means = np.array([0.3, 0.4, 0.8, 0.2, 1.0]) # g2
        self.cov = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]

        # Separate trackers for each regret type
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
        plt.title("Cumulative Regret Over Time (CUSUM CON-LCB)")
        plt.xlabel("Time Step (t)")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.grid(True)
        plt.show()

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

if __name__ == "__main__":
    
    T = 50000
    K = 4
    THRESHOLD1 = 0.5
    THRESHOLD2 = 0.4
    
    M = 50       # Min samples before starting detector
    epsilon = 0.1 # Drift tolerance
    h = 50       # Detection threshold

    print("--- CUSUM CON-LCB ---")
    print(f"Time Horizon (T): {T}")
    print(f"Number of Arms (K): {K}")
    print(f"Constraint 1 (g1_mean) <= {THRESHOLD1}")
    print(f"Constraint 2 (g2_mean) <= {THRESHOLD2}")
    print(f"Note: Environment changes at T/2.")
    print(f"CUSUM params: M={M}, epsilon={epsilon}, h={h}")

    feasibility_flag, regret_histories = cusum_con_lcb(T, K, THRESHOLD1, THRESHOLD2, M, epsilon, h)

    print("\n--- Simulation Results ---")
    print(f"{'Instance was determined to be Feasible' if feasibility_flag == 1 else 'Instance was determined to be Infeasible'}")
    
    if regret_histories["suboptimality"]:
        print(f"Final Suboptimality Regret: {regret_histories['suboptimality'][-1]:.2f}")
    if regret_histories["infeasibility"]:
        print(f"Final Infeasibility Regret: {regret_histories['infeasibility'][-1]:.2f}")
