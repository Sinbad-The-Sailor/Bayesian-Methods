import pandas as pd
import numpy as np

from scipy.stats import chi2, norm
from matplotlib import pyplot as plt


rain_data = pd.read_csv('data/Percipitation.csv')
rain_data.drop(columns=['ind'], inplace=True)
rain_data_log_transformed = np.log(rain_data)
y = np.array(rain_data_log_transformed)

# Prior hyperparameters.
kappa_0 = 1
mu_0 = 0
nu_0 = 0
sigma_0 = 1
n = len(rain_data)

# Nr simulations
N_SIMULATIONS = 25


def scale_inv_chi2_rvs(nu: float, sigma: float) -> float:
    return nu * sigma / chi2.rvs(nu)


def gibbs_sample_joint_posterior(size: int = 20) -> list:
    samples = []
    sim_mu = 1

    kappa_n = kappa_0 + n
    mu_n = kappa_0 * mu_0 / kappa_n + n * np.mean(y) / kappa_n
    nu_n = nu_0 + len(rain_data)

    for _ in range(size):
        # Draw posterior sigma
        sigma_n = (nu_0 * sigma_0 + (y-sim_mu).T @ (y-sim_mu))/nu_n
        sim_sigma = scale_inv_chi2_rvs(nu_n, sigma_n)

        samples.append((float(sim_mu), float(sim_sigma)))

        # Draw posterior mu
        tau_n = sim_sigma / kappa_n
        sim_mu = norm.rvs(mu_n, np.sqrt(tau_n))

        samples.append((float(sim_mu), float(sim_sigma)))

    return samples


if __name__ == '__main__':
    samples = gibbs_sample_joint_posterior(N_SIMULATIONS)
    print(samples)
    x = [sample[0] for sample in samples]
    y = [sample[1] for sample in samples]
    plt.plot(x, y, alpha=0.5)
    plt.plot(x[0], y[0], marker='s', color='green')
    plt.plot(x[-1], y[-1], marker='o', color='green')
    plt.show()
