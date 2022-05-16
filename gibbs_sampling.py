from random import sample
import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy.stats import chi2, norm
from matplotlib import pyplot as plt

plt.style.use(['science', 'notebook', 'grid'])

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
N_SIMULATIONS = 250


def scale_inv_chi2_rvs(nu: float, sigma: float) -> float:
    return nu * sigma / chi2.rvs(nu)


def gibbs_sample_joint_posterior(size: int = 20) -> list:
    plot_samples = []
    sim_mu = 1

    kappa_n = kappa_0 + n
    mu_n = kappa_0 * mu_0 / kappa_n + n * np.mean(y) / kappa_n
    nu_n = nu_0 + len(rain_data)

    for _ in range(size):

        # Draw posterior sigma
        sigma_n = (nu_0 * sigma_0 + (y-sim_mu).T @ (y-sim_mu))/nu_n
        sim_sigma = scale_inv_chi2_rvs(nu_n, sigma_n)
        plot_samples.append((float(sim_mu), float(sim_sigma)))

        # Draw posterior mu
        tau_n = sim_sigma / kappa_n
        sim_mu = norm.rvs(mu_n, np.sqrt(tau_n))
        plot_samples.append((float(sim_mu), float(sim_sigma)))

    return plot_samples


if __name__ == '__main__':
    plot_samples = gibbs_sample_joint_posterior(N_SIMULATIONS)
    x = [sample[0] for sample in plot_samples]
    y = [sample[1] for sample in plot_samples]

    sampled_mu = x[::2]
    sampled_sigma = y[::2]
    n_iterations = np.array([i for i in range(N_SIMULATIONS)])

    mu_efficiency = np.round(1 + 2 * np.sum(sm.tsa.acf(sampled_mu)), 2)
    sigma_efficiency = np.round(1 + 2 * np.sum(sm.tsa.acf(sampled_sigma)), 2)

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(x, y, alpha=0.7)
    axs[0].plot(x[0], y[0], marker='s', color='green')
    axs[0].plot(x[-1], y[-1], marker='o', color='green')
    axs[0].set_title('Gibbs Path')

    axs[1].plot(n_iterations, x[::2], label='IF: '+str(mu_efficiency))
    axs[1].set_title('$\mu$ sampling')
    axs[1].legend()
    axs[2].plot(n_iterations, y[::2], label='IF: '+str(sigma_efficiency))
    axs[2].set_title('$\sigma^2$ sampling')
    axs[2].legend()
    plt.show()

    # Sampling from posterior distribtion.
    sampled_predicted_y = []
    for i in range(N_SIMULATIONS):
        sampled_predicted_y.append(
            norm.rvs(sampled_mu[i], np.sqrt(sampled_sigma[i])))

    fig, axs = plt.subplots(1, 2)
    axs[0].hist(np.exp(y), edgecolor='black',
                alpha=0.7, label='Data', bins=80, color='green')
    axs[0].legend()
    axs[1].hist(np.exp(sampled_predicted_y), bins=80, edgecolor='black',
                alpha=0.7, label='Posterior prediction')
    axs[1].legend()
    plt.show()
