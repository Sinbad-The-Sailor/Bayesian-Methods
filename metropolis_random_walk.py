import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging

from scipy.stats import multivariate_normal, bernoulli
from scipy.optimize import minimize
from matplotlib import pyplot as plt


bid_data = np.loadtxt('data/eBayNumberOfBidderData.dat', unpack=True)
columns = ['nBids', 'Const', 'PowerSeller', 'VerifyID',  'Sealed',  'Minblem',
           'MajBlem', 'LargNeg', 'LogBook', 'MinBidShare']
df = pd.DataFrame(bid_data.T, columns=columns)


def log_prior(x: np.array, design_matrix: np.array) -> float:
    epsilon = 1e-16
    mean = np.zeros(len(x))
    covariance_matrix = np.linalg.inv((design_matrix.T @ design_matrix))
    return np.log(multivariate_normal.pdf(x, mean, covariance_matrix) + epsilon)


def log_likelihood(w: np.array, X: np.array, t: np.array) -> float:
    return np.sum(t @ X @ w - np.exp(X @ w))


def objective(w: np.array, X: np.array, t: np.array) -> float:
    return -(log_prior(w, X) + log_likelihood(w, X, t))


class MetropolisRandomWalk:
    SAMPLE_SIZE = 5000

    def __init__(self, posterior_functiuon, init_sample, step, cov_matrix):
        self.posterior_function = posterior_functiuon
        self.init_sample = init_sample
        self.step = step
        self.cov_matrix = cov_matrix
        self.samples = None

    def _draw_candidate_sample(self, prev_sample: np.array) -> np.array:
        return multivariate_normal.rvs(prev_sample, self.step * self.cov_matrix)

    def _calculate_alpha(self, candidate_sample: np.array, prev_sample: np.array) -> float:
        '''
        Soruce: http://www.kris-nimark.net/pdf/M-HSlides.pdf
        '''
        alpha = min((self.posterior_function(candidate_sample)) -
                    (self.posterior_function(prev_sample)), 0)
        return np.exp(alpha)

    def generate_samples(self, sample_size: int = SAMPLE_SIZE) -> list:
        samples = []
        prev_sample = self.init_sample

        for _ in range(self.SAMPLE_SIZE):
            candidate_sample = self._draw_candidate_sample(
                prev_sample=prev_sample)
            alpha = self._calculate_alpha(
                candidate_sample=candidate_sample, prev_sample=prev_sample)

            decision = bernoulli.rvs(alpha)
            if decision:
                prev_sample = candidate_sample
            samples.append(prev_sample)
        self.samples = samples
        return samples

    def plot_simulations(self):
        # TODO: remove hard coded fig grid.
        fig_grid = (2, 4)
        fig, axs = plt.subplots(fig_grid[0], fig_grid[1])
        x = [i for i in range(self.SAMPLE_SIZE)]
        for j in range(4):
            y = [sample[j] for sample in self.samples]
            axs[0, j].plot(x, y)
        for j in range(4, 8):
            y = [sample[j] for sample in self.samples]
            axs[1, j-4].plot(x, y)
        plt.show()


if __name__ == '__main__':
    poisson_regression = sm.GLM(df['nBids'],
                                df.drop(columns=['nBids']),
                                family=sm.families.Poisson()
                                )

    poisson_results = poisson_regression.fit()
    print(poisson_results.summary())
    mode_params = poisson_results.params

    # Data matrices.
    target = df['nBids']
    design_matrix = df.drop(columns=['nBids'])

    # Initial parameter distribtuion.
    mean = np.zeros(len(mode_params))
    cov = np.linalg.inv(design_matrix.T @ design_matrix)

    # Constant for optimzation tries.
    MAX_NUMBER_OF_OPTIMIZATION_ITERATIONS = 20

    iteration = 0
    success = False
    while not success:
        iteration = iteration + 1
        x0 = multivariate_normal.rvs(mean, cov, 1)
        solution = minimize(objective, x0=x0, args=(
            design_matrix, target), method='BFGS')

        # Check for successful and resonable optimzation.
        success = solution.success
        if iteration > MAX_NUMBER_OF_OPTIMIZATION_ITERATIONS:
            logging.critical(' Max number of iterations reached.')
            success = True

    mode_param_norm = solution.x
    mode_param_inv_hess = solution.hess_inv
    print(multivariate_normal.rvs(mode_param_norm, -mode_param_inv_hess, 1))

    # Creating ad-hoc posterior function to be used in metropolios walk.
    # Note: I had troubles with the -Hessian not being postive semi definite.
    # Using posterior proprortionallity and log of alpha for computational reasons.
    def test_posterior_function(x: np.array) -> float:
        return log_likelihood(x, design_matrix, target) + log_prior(x, design_matrix=design_matrix)

    # Creating a Metropolios Random Walk for multivariate normal appr. posterior.
    init_sample = np.zeros(9)
    metropolis_walk = MetropolisRandomWalk(
        test_posterior_function, init_sample=init_sample, step=0.1, cov_matrix=np.identity(9))
    print(metropolis_walk.generate_samples())
    metropolis_walk.plot_simulations()


class MetropolisRandomWalk_old:

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
