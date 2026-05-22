"""
algorithms.py  —  All bandit algorithms in one place.

Algorithms
----------
  SWConLCB          sliding-window CON-LCB   (w=300,  xi=3.0)
  DConLCB           discounted  CON-LCB      (g=0.995, xi=0.5)
  ConLCB            stationary  CON-LCB      (full history)
  EpsilonGreedy     decaying epsilon-greedy
"""
import numpy as np
from collections import deque

TAU_LOOSE = 0.30  # The higher priority "must not exceed" limit
TAU_TIGHT = 0.20  # The lower priority "nice to have" target
A         = np.ones(2)  

def _select(ucb_rewards, means, lcbs, tau_loose, tau_tight):
    K = means.shape[0]
    
    # Tier 0: Strict target met (<= 20%)
    tier0 = [k for k in range(K) if lcbs[k,1] <= tau_tight]
    if tier0:
        return int(tier0[np.argmax(ucb_rewards[tier0])]), 0

    # Tier 1: Relaxed to hard limit (<= 30%)
    tier1 = [k for k in range(K) if lcbs[k,1] <= tau_loose]
    if tier1:
        return int(tier1[np.argmax(ucb_rewards[tier1])]), 1

    # Tier 2: Fallback to the arm with the lowest observed resistance mean
    return int(np.argmin(means[:, 1])), 2

def _empty():
    return dict(cum_reward=0.0, cum_viol_loose=0.0, cum_viol_tight=0.0,
                reward_hist=[], viol_loose_hist=[], viol_tight_hist=[])

def _record(res, r_vec, tau_loose, tau_tight):
    res["cum_reward"] += r_vec[0]
    res["cum_viol_tight"] += max(0.0, r_vec[1] - tau_tight) # Penalty for missing 20%
    res["cum_viol_loose"] += max(0.0, r_vec[1] - tau_loose) # Penalty for missing 30%
    res["reward_hist"].append(res["cum_reward"])
    res["viol_tight_hist"].append(res["cum_viol_tight"])
    res["viol_loose_hist"].append(res["cum_viol_loose"])


# ── 1. SW-CON-LCB ─────────────────────────────────────────────────────────────
class SWConLCB:
    def __init__(self, w=400, tau_loose=TAU_LOOSE, tau_tight=TAU_TIGHT, xi=0.01, a=A):
        self.w, self.tau_loose, self.tau_tight = w, tau_loose, tau_tight
        self.xi, self.a = xi, np.array(a, float)

    def run(self, env):
        K, T, w = env.K, env.T, self.w
        window   = deque(maxlen=w)
        sums     = np.zeros((K, 2))
        n        = np.zeros(K)
        res      = _empty()

        def _push(arm, r_vec):
            if len(window) == w:
                ea, er = window[0]
                sums[ea] -= er;  n[ea] -= 1
            window.append((arm, r_vec))
            sums[arm] += r_vec;  n[arm] += 1

        for arm in range(K):
            r_vec = env.step(arm)
            _push(arm, r_vec)
            _record(res, r_vec, self.tau_loose, self.tau_tight)

        for t in range(K, T):
            valid = n > 0
            means = np.where(valid[:,None], sums / np.maximum(n[:,None], 1e-9), 0.0)
            
            lcbs  = np.full((K, 2), -np.inf)
            ucb_rewards = np.full(K, np.inf)

            if valid.any():
                log_t = self.xi * np.log(max(min(t, w), 1))
                bonus = np.sqrt(log_t / (self.a * np.maximum(n[:, None], 1e-9)))
                
                lcbs[valid] = means[valid] - bonus[valid]
                ucb_rewards[valid] = means[valid, 0] + bonus[valid, 0]
                
            arm, _ = _select(ucb_rewards, means, lcbs, self.tau_loose, self.tau_tight)
            r_vec  = env.step(arm)
            _push(arm, r_vec)
            _record(res, r_vec, self.tau_loose, self.tau_tight)
        return res


# ── 2. D-CON-LCB ──────────────────────────────────────────────────────────────
class DConLCB:
    def __init__(self, gamma=0.995, tau_loose=TAU_LOOSE, tau_tight=TAU_TIGHT, xi=0.0075, a=A):
        self.gamma, self.tau_loose, self.tau_tight = gamma, tau_loose, tau_tight
        self.xi, self.a = xi, np.array(a, float)

    def run(self, env):
        K, T   = env.K, env.T
        S      = np.zeros((K, 2))
        N      = np.zeros(K)
        eps    = 1e-9
        res    = _empty()

        for arm in range(K):
            S *= self.gamma;  N *= self.gamma
            r_vec = env.step(arm)
            S[arm] += r_vec;  N[arm] += 1.0
            _record(res, r_vec, self.tau_loose, self.tau_tight)

        for _ in range(K, T):
            means    = S / (N[:,None] + eps)
            beta     = N.sum()
            log_term = self.xi * np.log(2.0 * beta**2 + 1.0)
            bonus    = np.sqrt(log_term / (self.a * (N[:,None] + eps)))
            
            lcbs     = means - bonus
            ucb_rewards = means[:, 0] + bonus[:, 0]
            
            arm, _   = _select(ucb_rewards, means, lcbs, self.tau_loose, self.tau_tight)

            S *= self.gamma;  N *= self.gamma
            r_vec  = env.step(arm)
            S[arm] += r_vec;  N[arm] += 1.0
            _record(res, r_vec, self.tau_loose, self.tau_tight)
        return res


# ── 3. Stationary CON-LCB ─────────────────────────────────────────────────────
class ConLCB:
    def __init__(self, tau_loose=TAU_LOOSE, tau_tight=TAU_TIGHT, xi=2.0, a=A):
        self.tau_loose, self.tau_tight = tau_loose, tau_tight
        self.xi, self.a = xi, np.array(a, float)

    def run(self, env):
        K, T   = env.K, env.T
        sums   = np.zeros((K, 2))
        n      = np.zeros(K)
        res    = _empty()

        for arm in range(K):
            r_vec = env.step(arm)
            sums[arm] += r_vec;  n[arm] += 1
            _record(res, r_vec, self.tau_loose, self.tau_tight)

        for t in range(K, T):
            means = sums / n[:,None]
            bonus = np.sqrt(self.xi * np.log(max(t,1)) / (self.a * n[:,None]))
            
            lcbs  = means - bonus
            ucb_rewards = means[:, 0] + bonus[:, 0]
            
            arm, _ = _select(ucb_rewards, means, lcbs, self.tau_loose, self.tau_tight)
            
            r_vec  = env.step(arm)
            sums[arm] += r_vec;  n[arm] += 1
            _record(res, r_vec, self.tau_loose, self.tau_tight)
        return res


# ── 4. Epsilon-Greedy ─────────────────────────────────────────────────────────
class EpsilonGreedy:
    def __init__(self, tau_loose=TAU_LOOSE, tau_tight=TAU_TIGHT, c=2.0):
        self.tau_loose, self.tau_tight, self.c = tau_loose, tau_tight, c

    def run(self, env):
        K, T   = env.K, env.T
        rng = np.random.default_rng(env._rng.integers(2**32))
        sums   = np.zeros((K, 2))
        n      = np.zeros(K)
        res    = _empty()

        for arm in range(K):
            r_vec = env.step(arm)
            sums[arm] += r_vec;  n[arm] += 1
            _record(res, r_vec, self.tau_loose, self.tau_tight)

        for t in range(K, T):
            if rng.random() < min(0.1, self.c * K / (t + 1)):
                arm = int(rng.integers(K))
            else:
                means    = sums / n[:,None]
                tier0 = [k for k in range(K) if means[k,1] <= self.tau_tight]
                
                if tier0:
                    pool = tier0
                else:
                    tier1 = [k for k in range(K) if means[k,1] <= self.tau_loose]
                    pool  = tier1 if tier1 else [int(np.argmin(means[:, 1]))]
                arm  = int(pool[np.argmax([means[k,0] for k in pool])])
                
            r_vec = env.step(arm)
            sums[arm] += r_vec;  n[arm] += 1
            _record(res, r_vec, self.tau_loose, self.tau_tight)
        return res