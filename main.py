import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from environment import ReplayEnv
from algorithms  import SWConLCB, DConLCB, ConLCB, EpsilonGreedy

# ── config ─────────────────────────────────────────────────
CFG = dict(
    rounds_per_quarter = 200,
    max_quarters       = 20,
    tau_loose          = 0.30,
    tau_tight          = 0.20,
)

ALGORITHMS = {
    "SW-CON-LCB" : SWConLCB (w=400,  xi=0.01,  tau_loose=CFG["tau_loose"], tau_tight=CFG["tau_tight"]),
    "D-CON-LCB"  : DConLCB  (gamma=0.995, xi=0.0075, tau_loose=CFG["tau_loose"], tau_tight=CFG["tau_tight"]),
    "CON-LCB"    : ConLCB   (xi=2.0, tau_loose=CFG["tau_loose"], tau_tight=CFG["tau_tight"]),
    "EpsilonGreedy": EpsilonGreedy(tau_loose=CFG["tau_loose"], tau_tight=CFG["tau_tight"]),
}

COLORS = {
    "SW-CON-LCB"   : "#2196F3",
    "D-CON-LCB"    : "#00087B",
    "CON-LCB"      : "#d4d4d4",
    "EpsilonGreedy": "#4e4c4c",
}

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_optimal_expected_rewards(env_cfg):
    env = ReplayEnv(env_cfg["rounds_per_quarter"], env_cfg["max_quarters"])
    opt_r = np.zeros(env.T)
    
    for t in range(env.T):
        q_idx = min(t // env.Tb, len(env.quarters) - 1)
        quarter = env.quarters[q_idx]

        tier0_best_r = -np.inf 
        tier1_best_r = -np.inf 
        fallback_arm = None
        lowest_res   = np.inf

        for arm in range(env.K):
            pool      = env.pools.get((quarter, arm)) or env._nearest(arm, quarter)
            arm_means = np.mean(pool, axis=0)[:2] 
            
            res_mean = arm_means[1]
            rew_mean = arm_means[0]

            # Tier 0: Meets strict 20% target
            if res_mean <= env_cfg["tau_tight"]:
                if rew_mean > tier0_best_r:
                    tier0_best_r = rew_mean
            # Tier 1: Fails 20%, but meets 30% hard limit
            elif res_mean <= env_cfg["tau_loose"]:
                if rew_mean > tier1_best_r:
                    tier1_best_r = rew_mean

            # Track Fallback
            if res_mean < lowest_res:
                lowest_res   = res_mean
                fallback_arm = arm

        if tier0_best_r > -np.inf:
            opt_r[t] = tier0_best_r
        elif tier1_best_r > -np.inf:
            opt_r[t] = tier1_best_r
        else:
            pool     = env.pools.get((quarter, fallback_arm)) or env._nearest(fallback_arm, quarter)
            opt_r[t] = np.mean(pool, axis=0)[0]
        
    return opt_r

# ── run ───────────────────────────────────────────────────────────────────────
def run_all(n_seeds):
    results = {}

    for name, alg in ALGORITHMS.items():
        print(f"\n{name}")
        seed_res = []
        for seed in tqdm(range(n_seeds)):
            env = ReplayEnv(CFG["rounds_per_quarter"], CFG["max_quarters"])
            env.reset(seed)
            seed_res.append(alg.run(env))
        results[name] = seed_res

        T = len(seed_res[0]["reward_hist"])
        avg_r  = np.mean([r["reward_hist"] for r in seed_res], axis=0)
        avg_vt = np.mean([r["viol_tight_hist"]  for r in seed_res], axis=0)
        avg_vl = np.mean([r["viol_loose_hist"]  for r in seed_res], axis=0)
        
        print(f"  reward={avg_r[-1]:.1f}  "
              f"viol_tight(20%)={avg_vt[-1]:.1f}  viol_loose(30%)={avg_vl[-1]:.1f}")

    return results

# ── plots ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 24,
    'font.family': 'serif',
    'mathtext.fontset': 'cm', 
})

def remove_box(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

def _thin(arr, n=500):
    step = max(1, len(arr) // n)
    return np.arange(1, len(arr)+1)[::step], arr[::step]

def plot_regret(results):
    opt_r = get_optimal_expected_rewards(CFG)
    opt_cum = np.cumsum(opt_r)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, seed_res in results.items():
        regrets = [opt_cum - r["reward_hist"] for r in seed_res]
        avg = np.mean(regrets, axis=0)
        std = np.std(regrets, axis=0)
        
        t, y = _thin(avg)
        _, s = _thin(std)
        ax.plot(t, y, color=COLORS[name], lw=2.5, label=name)
        ax.fill_between(t, y-s, y+s, color=COLORS[name], alpha=0.15)
        
    ax.set_xlabel(r"Time Step ($t$)", fontsize=26)
    ax.set_ylabel(r"Cumulative Regret $R_T^{sub}$")
    
    remove_box(ax)
    ax.legend(fontsize=16, frameon=False)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / "regret.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_violations(results):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for name, seed_res in results.items():
        for ax, key in zip(axes, ["viol_tight_hist", "viol_loose_hist"]):
            avg = np.mean([r[key] for r in seed_res], axis=0)
            t, y = _thin(avg)
            ax.plot(t, y, color=COLORS[name], lw=2.5, label=name)

    ylabels = [r"Target Violation ($\tau=0.2$)", r"Limit Violation ($\tau=0.3$)"]
    for ax, ylabel in zip(axes, ylabels):
        ax.set_xlabel(r"Time Step ($t$)", fontsize=26)
        ax.set_ylabel(ylabel)
        
        remove_box(ax)
        ax.legend(fontsize=16, frameon=False)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_DIR / "violations.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_violation_rates(results):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for name, seed_res in results.items():
        for ax, key in zip(axes, ["viol_tight_hist", "viol_loose_hist"]):
            avg = np.mean([r[key] for r in seed_res], axis=0)
            T   = len(avg)
            rate = avg / np.arange(1, T+1)
            t, y = _thin(rate)
            ax.plot(t, y, color=COLORS[name], lw=2.5, label=name)

    thresholds = [CFG["tau_tight"], CFG["tau_loose"]]
    titles     = [r"Target Violation Rate ($\tau=0.2$)", r"Limit Violation Rate ($\tau=0.3$)"]
    
    for ax, tau, title in zip(axes, thresholds, titles):
        ax.axhline(tau, color="black", ls="--", lw=2.0, alpha=0.5, label=f"$\\tau={tau}$")
        ax.set_xlabel(r"Time Step ($t$)", fontsize=26)
        ax.set_ylabel(title)
        
        remove_box(ax)
        ax.legend(fontsize=16, frameon=False)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_DIR / "violation_rates.pdf", bbox_inches="tight")
    plt.close(fig)

def save_summary(results):
    rows = []
    for name, seed_res in results.items():
        T      = len(seed_res[0]["reward_hist"])
        avg_r  = np.mean([r["cum_reward"] for r in seed_res])
        avg_vt = np.mean([r["cum_viol_tight"]  for r in seed_res])
        avg_vl = np.mean([r["cum_viol_loose"]  for r in seed_res])
        rows.append(dict(
            algorithm    = name,
            cum_reward   = round(avg_r,  2),
            cum_viol_t   = round(avg_vt, 2),
            cum_viol_l   = round(avg_vl, 2),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nSummary:\n{df.to_string(index=False)}")

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    results = run_all(args.seeds)
    plot_regret(results)
    plot_violations(results)
    plot_violation_rates(results)
    save_summary(results)