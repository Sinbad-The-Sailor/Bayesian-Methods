import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging

from scipy.stats import multivariate_normal, bernoulli, poisson
from scipy.optimize import minimize
from matplotlib import pyplot as plt

plt.style.use(['science', 'notebook', 'grid'])


bid_data = np.loadtxt('data/eBayNumberOfBidderData.dat', unpack=True)
columns = ['nBids', 'Const', 'PowerSeller', 'VerifyID',  'Sealed',  'Minblem',
           'MajBlem', 'LargNeg', 'LogBook', 'MinBidShare']
df = pd.DataFrame(bid_data.T, columns=columns)


def log_prior(x: np.array, design_matrix: np.array) -> float:
    epsilon = 1e-16
    mean = np.zeros(len(x))
    covariance_matrix = 100 * np.linalg.inv((design_matrix.T @ design_matrix))
    return np.log(multivariate_normal.pdf(x, mean, covariance_matrix) + epsilon)


def log_likelihood(w: np.array, X: np.array, t: np.array) -> float:
    return np.sum(t @ X @ w - np.exp(X @ w))


def objective(w: np.array, X: np.array, t: np.array) -> float:
    return -(log_prior(w, X) + log_likelihood(w, X, t))


def posterior_predictive_distribution(x: np.array, w: np.array) -> float:
    mean = x.T @ w
    return poisson.rvs(mean, size=1)[0]


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
        print(np.exp(alpha))
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
        # TODO: remove hard-coded fig grid.
        fig_grid = (3, 3)
        fig, axs = plt.subplots(fig_grid[0], fig_grid[1])
        x = [i for i in range(self.SAMPLE_SIZE)]
        for j in range(3):
            y = [sample[j] for sample in self.samples]
            axs[0, j].plot(x, y)
        for j in range(3, 6):
            y = [sample[j] for sample in self.samples]
            axs[1, j-3].plot(x, y)
        for j in range(6, 9):
            y = [sample[j] for sample in self.samples]
            axs[2, j-6].plot(x, y)
        plt.show()


if __name__ == '__main__':
    poisson_regression = sm.GLM(df['nBids'],
                                df.drop(columns=['nBids']),
                                family=sm.families.Poisson()
                                )

    poisson_results = poisson_regression.fit()
    print(poisson_results.summary())
    mode_params = poisson_results.params
    pos_inv_hess = np.linalg.inv(poisson_regression.hessian(mode_params))
    print('pos hess')
    print(pos_inv_hess)

    # Data matrices.
    target = df['nBids']
    design_matrix = df.drop(columns=['nBids'])

    # Initial parameter distribtuion.
    mean = np.zeros(len(mode_params))
    mean = mode_params
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

    def test_posterior_function(x: np.array) -> float:
        return log_likelihood(x, design_matrix, target) + log_prior(x, design_matrix=design_matrix)

    # Creating a Metropolios Random Walk for multivariate normal appr. posterior.
    init_sample = mode_param_norm
    metropolis_walk = MetropolisRandomWalk(
        test_posterior_function, init_sample=init_sample, step=0.001, cov_matrix=-pos_inv_hess)
    posterior_samples = metropolis_walk.generate_samples()
    # metropolis_walk.plot_simulations()

    example_data = np.array([1, 1, 0, 1, 0, 1, 0, 1.2, 0.8])
    pred_posterior = [posterior_predictive_distribution(
        example_data, w) for w in posterior_samples]

    plt.hist(pred_posterior, edgecolor='black')
    plt.show()
