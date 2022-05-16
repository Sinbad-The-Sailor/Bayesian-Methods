import numpy as np
import pandas as pd
import statsmodels.api as sm


bid_data = np.loadtxt('data/eBayNumberOfBidderData.dat', unpack=True)
columns = ['nBids', 'Const', 'PowerSeller', 'VerifyID',  'Sealed',  'Minblem',
           'MajBlem', 'LargNeg', 'LogBook', 'MinBidShare']
df = pd.DataFrame(bid_data.T, columns=columns)

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


class MetropolisRandomWalk:
    def __init__(self, posterior_function: function, init_sample: np.array):
        pass
