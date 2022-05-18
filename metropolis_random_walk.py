import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging

from scipy.stats import multivariate_normal, bernoulli

bid_data = np.loadtxt('data/eBayNumberOfBidderData.dat', unpack=True)
columns = ['nBids', 'Const', 'PowerSeller', 'VerifyID',  'Sealed',  'Minblem',
           'MajBlem', 'LargNeg', 'LogBook', 'MinBidShare']
df = pd.DataFrame(bid_data.T, columns=columns)


def my_posterior(params: np.array) -> float:
    return multivariate_normal.pdf(params, poisson_regression.hessian(params))


class MetropolisRandomWalk:
    def __init__(self, posterior_function, init_sample: np.array, step_lenght: float, covariance_matrix: np.array):
        self.posterior_function = posterior_function
        self.init_sample = init_sample
        self.step_lenght = step_lenght
        self.covariance_matrix = step_lenght * covariance_matrix

    def simulate_params(self, size: int = 100) -> list:
        samples = []
        prev_sample = self.init_sample

        for _ in range(size):
            current_sample = multivariate_normal.rvs(prev_sample,
                                                     self.covariance_matrix,
                                                     1
                                                     )
            try:
                beta = self.posterior_function(
                    current_sample)/self.posterior_function(prev_sample)
            except TypeError as e:
                logging.critical(e, exc_info=True)
            except ValueError as e:
                logging.critical(e, exc_info=True)
            finally:
                print('H')

            alpha = np.min(1, beta)
            decision = bernoulli.rvs(alpha, size=1)
            if decision:
                current_sample = prev_sample

            samples.append(current_sample)

        return samples


if __name__ == '__main__':
    poisson_regression = sm.GLM(df['nBids'],
                                df.drop(columns=['nBids']),
                                family=sm.families.Poisson()
                                )

    poisson_results = poisson_regression.fit()
    print(poisson_results.summary())
    mode_params = poisson_results.params
    mode_hessian = poisson_regression.hessian(mode_params)
    appr_posterior_cov_matrix = np.linalg.inv(mode_hessian)
    print(appr_posterior_cov_matrix)

    init_sample = np.ones(9)
    metropolis_sampler = MetropolisRandomWalk(
        my_posterior, init_sample=init_sample, step_lenght=1, covariance_matrix=mode_hessian)
    samples = metropolis_sampler.simulate_params()
