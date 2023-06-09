import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import copy

from scipy.stats import bartlett
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

os.chdir('H:/I.R/Project：Neuroscience/Data Science Training')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)

input_path = './Training dataset/EFA+CFA/FactorAnalysis.csv'
data = pd.read_csv(input_path)
variable_names = data.columns
data.head(6)

data.iloc[0]

Zdata = pd.DataFrame(StandardScaler().fit_transform(data))
corrcoef = Zdata.corr()
corr = copy.deepcopy(corrcoef)

def kmo(dataset_corr):
    corr_inv = np.linalg.inv(dataset_corr)
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    A = np.ones((nrow_inv_corr, ncol_inv_corr))
    for i in range(nrow_inv_corr):
        for j in range(i, ncol_inv_corr):
            A[i, j] = -(corr_inv[i, j]) / (np.sqrt(corr_inv[i, i] * corr_inv[j, j]))
            A[j, i] = A[i, j]
    dataset_corr = np.asarray(dataset_corr)
    kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
    kmo_denom = kmo_num + np.sum(np.square(np.triu(dataset_corr)))
    kmo_value = kmo_num / kmo_denom
    return kmo_value

kmo_value, kmo_model = calculate_kmo(corr)
kmo_value

def scree_plot(eigenvalues):
    plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2)
    plt.xlabel('Number of Factors')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.axhline(1, color='r', linestyle='--')
    plt.show()

def run_factor_analysis(data, num_factors):
    fa = FactorAnalysis(n_components=num_factors, random_state=0)
    fa.fit(data)
    factor_loadings = fa.components_.T
    eigenvalues = fa.noise_variance_ + fa.latent_vars_
    return factor_loadings, eigenvalues

eigenvalues, factor_loadings = run_factor_analysis(corr, 3)
scree_plot(eigenvalues)

def interpret_factors(factor_loadings, variable_names):
    factor_loadings_df = pd.DataFrame(factor_loadings, columns=['Factor 1', 'Factor 2', 'Factor 3'])
    factor_loadings_df['Variable'] = variable_names
    return factor_loadings_df

factor_loadings_df = interpret_factors(factor_loadings, variable_names)
factor_loadings_df

def perform_bartlett_test(corr, sample_size):
    chi2, p_value = bartlett(*corr)
    df = sample_size * (sample_size - 1) / 2
    return chi2, p_value, df

sample_size = len(data)
chi2, p_value, df = perform_bartlett_test(corr, sample_size)
chi2, p_value, df

