import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy import stats
import arviz as az

def start():
    # Load the dataset
    df = pd.read_csv("cookie_cats.csv")

    # Map versions to A and B
    # 'gate_30' is control (A), 'gate_40' is treatment (B)
    df['group'] = df['version'].map({'gate_30': 'A', 'gate_40': 'B'})

    # -------------------------------
    # Function for Bayesian A/B test
    # -------------------------------
    def bayesian_ab_test(obs_A, obs_B, label_A="A", label_B="B", metric="retention"):
        with pm.Model() as model:
            p_A = pm.Uniform("p_A", 0, 1)
            p_B = pm.Uniform("p_B", 0, 1)
            delta = pm.Deterministic("delta", p_A - p_B)
            
            obsA = pm.Bernoulli("obs_A", p=p_A, observed=obs_A)
            obsB = pm.Bernoulli("obs_B", p=p_B, observed=obs_B)
            
            step = pm.Metropolis()
            trace = pm.sample(100, step=step, chains=2, progressbar=True)

        p_A_samples = np.concatenate(trace.posterior.p_A.data[:, 50:])
        p_B_samples = np.concatenate(trace.posterior.p_B.data[:, 50:])
        delta_samples = np.concatenate(trace.posterior.delta.data[:, 50:])
        
        # Plot posteriors
        plt.figure(figsize=(12, 10))

        plt.subplot(311)
        plt.hist(p_A_samples, bins=30, alpha=0.7, label=f"{label_A} posterior", color="blue", density=True)
        plt.title(f"Posterior of {label_A} - {metric}")
        plt.legend()

        plt.subplot(312)
        plt.hist(p_B_samples, bins=30, alpha=0.7, label=f"{label_B} posterior", color="green", density=True)
        plt.title(f"Posterior of {label_B} - {metric}")
        plt.legend()

        plt.subplot(313)
        plt.hist(delta_samples, bins=30, alpha=0.7, label="Delta (A - B)", color="purple", density=True)
        plt.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.title("Posterior of Delta")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Report probability
        print(f"Probability {label_A} is WORSE than {label_B}: {np.mean(delta_samples < 0):.3f}")
        print(f"Probability {label_A} is BETTER than {label_B}: {np.mean(delta_samples > 0):.3f}")
        return np.mean(delta_samples > 0), np.mean(delta_samples < 0)

    # ---------------------------------
    # 1. Test on 1-day retention
    # ---------------------------------
    print("------ 1-Day Retention ------")
    ret1_A = df[df.group == 'A']['retention_1'].values
    ret1_B = df[df.group == 'B']['retention_1'].values
    bayesian_ab_test(ret1_A, ret1_B, label_A="gate_30", label_B="gate_40", metric="1-day retention")

    # ---------------------------------
    # 2. Test on 7-day retention
    # ---------------------------------
    print("\n------ 7-Day Retention ------")
    ret7_A = df[df.group == 'A']['retention_7'].values
    ret7_B = df[df.group == 'B']['retention_7'].values
    bayesian_ab_test(ret7_A, ret7_B, label_A="gate_30", label_B="gate_40", metric="7-day retention")

if __name__ == '__main__':
    start()