import pandas as pd
import numpy as np

from scipy.stats import chi2, norm


rain_data = pd.read_csv('data/Percipitation.csv')
rain_data.drop(columns=['ind'], inplace=True)
rain_data_log_transformed = np.log(rain_data)

# Prior hyperparameters.
mu_0 = 0
tau_0 = 0
nu_0 = 0
sigma_0 = 0

# Posterior parameters.
nu_n = nu_0 + len(rain_data)


def scale_inv_chi2_rvs(nu: float, sigma: float) -> float:
    return nu * sigma / chi2.rvs(nu)


def gibbs_sample_joint_posterior(paramas: np.array, size: int = 20) -> list:
    pass
