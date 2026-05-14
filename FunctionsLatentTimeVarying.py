from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer

import pymc as pm
import arviz as az
import pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from collections import defaultdict
from math import log2
from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from math import log2


def load_year_data(file_path, start_year, end_year):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Create list of year columns as integers
    year_columns = list(range(start_year, end_year + 1))

    # Columns to keep: fixed ones + available year columns
    fixed_columns = ['Country', 'Variable']
    selected_columns = fixed_columns + [col for col in year_columns if col in df.columns]

    # Return filtered DataFrame
    return df[selected_columns]

def load_panel_data_pov_gap(df, year_columns):

    # Preserve original country order
    original_country_order = df['Country'].drop_duplicates().tolist()
    df['Country'] = pd.Categorical(df['Country'], categories=original_country_order, ordered=True)

    id_vars = ['Country', 'Variable']
    df_long = df.melt(id_vars=id_vars, value_vars=year_columns,
                      var_name='Year', value_name='Value')

    df_pivot = df_long.pivot_table(index=['Country', 'Year'],
                                   columns='Variable', values='Value',
                                   sort=False).reset_index()

    df_pivot.columns.name = None
    df_pivot.columns = df_pivot.columns.str.strip()

    return df_pivot

def load_panel_data(file_path, year_columns):
    df = pd.read_excel(file_path, engine='openpyxl')

    # Preserve original country order
    original_country_order = df['Country'].drop_duplicates().tolist()
    df['Country'] = pd.Categorical(df['Country'], categories=original_country_order, ordered=True)

    id_vars = ['Country', 'Variable']
    df_long = df.melt(id_vars=id_vars, value_vars=year_columns,
                      var_name='Year', value_name='Value')

    df_pivot = df_long.pivot_table(index=['Country', 'Year'],
                                   columns='Variable', values='Value',
                                   sort=False).reset_index()

    df_pivot.columns.name = None
    df_pivot.columns = df_pivot.columns.str.strip()

    return df_pivot


def prepare_regression_data(df):
    df_clean = df.dropna(subset=['GDP growth','Capital formation', 'Population growth'])

    X = df_clean[['Capital formation','Population growth']].values
    Y = df_clean['GDP growth'].values
    meta = df_clean[['Country', 'Year']]

    # Create a combined DataFrame for viewing
    combined_df = meta.copy()
    combined_df['Capital formation'] = df_clean['Capital formation'].values
    combined_df['Population growth'] = df_clean['Population growth'].values
    combined_df['GDP growth'] = df_clean['GDP growth'].values

    print(combined_df.head())  # print first few rows

    return X, Y, meta


def plot_panel_subplot(YV, data, subplot_position, title, xlim=(1971, 2017), country_labels=None):
    """
    Plots a panel time series in a given subplot.

    Parameters:
    - YV: 1D array of years (length T)
    - data: 2D array of shape (T, N), one column per country
    - subplot_position: tuple like (1, 3, 1) for subplot(1,3,1)
    - title: string, title of the subplot
    - xlim: tuple, x-axis limits
    - country_labels: optional list of country names for legend
    """
    plt.subplot(*subplot_position)
    T, N = data.shape
    for i in range(N):
        label = country_labels[i] if country_labels else None
        plt.plot(YV, data[:, i], label=label)
    plt.xlim(xlim)
    plt.title(title)
    if country_labels:
        plt.legend(fontsize='small')





def kernel(u, bandwidth):
    # Gaussian kernel
    return norm.pdf(u / bandwidth) / bandwidth

def epanechnikov_kernel(u):
    z = np.abs(u )
    return 0.75 * (1 - z**2) * (z <= 1)


def ksmooth(X, vt, Y, u, h):
    weights = epanechnikov_kernel((vt - u) / h)  # shape (n,)

    # Ensure weights are shape (n,)
    weights = np.asarray(weights).flatten()

    # Reshape X and Y if needed
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 2 and Y.shape[1] == 1:
        Y = Y.flatten()

    X_weighted = X * weights[:, np.newaxis]  # (n_samples, n_features)
    Y_weighted = Y * weights  # (n_samples,)

    XtWX = X.T @ X_weighted
    XtWY = X.T @ Y_weighted

    try:
        coef_fun = np.linalg.solve(XtWX, XtWY)
    except np.linalg.LinAlgError:
        coef_fun = np.linalg.pinv(XtWX) @ XtWY

    return coef_fun  # shape (n_features,)


def cv_ksmooth(X_list, V, Y, u, h):
    """
    Estimate coefficient function at each u using kernel smoothing.

    Parameters:
        X_list: list of (T, N) arrays, each representing a regressor
        V: (T, N) array of index variable
        Y: (T, N) array of outcomes
        u: (N,) array of evaluation points
        h: scalar bandwidth

    Returns:
        coef_fun: (D, N) array, where D is the number of regressors
    """
    T, N = V.shape
    D = len(X_list)
    coef_fun = np.zeros((D, N)) if D > 1 else np.zeros(N)

    for i in range(N):
        x = np.column_stack([X[:, i] for X in X_list]) if D > 1 else X_list[0][:, i].reshape(-1, 1)
        v = V[:, i]
        y = Y[:, i]
        result = ksmooth(x, v, y, u[i], h)

        if D > 1:
            coef_fun[:, i] = result
        else:
            coef_fun[i] = result

    return coef_fun


def find_coef(X_list,Y_panel, V, h,d):
    """
    Remove individual fixed effects from panel data using kernel-smoothed coefficients.

    Parameters:
    SX : ndarray (M, d)
        Regressor matrix (M = N * T).
    SY : ndarray (M,) or (M, 1)
        Dependent variable vector.
    vt : ndarray (T,)
        Scaled time points (e.g., np.linspace(1/T, 1, T)).
    h : float
        Bandwidth for kernel smoothing.

    Returns:
    modY : ndarray (M,)
        Dependent variable with fixed effects removed.
    """
    T, N = V.shape

    matrices = []

         # Kernel smoothing to estimate individual-specific time-varying betas
    for t in range(T):
         u = np.tile(((t + 1) / T), N)
         beta = cv_ksmooth(X_list, V, Y_panel,u, h)  # ksmooth needs to be defined
         matrices. append(beta)
    Ebeta_h = np.vstack(matrices)
    Ebeta = np.vstack([Ebeta_h[:, i].reshape(T, d) for i in range(N)])

    return Ebeta


def remove_fixed_effects(X_list,Y_panel,V,SX,SY, vt, h):
    """
    Remove individual fixed effects from panel data using kernel-smoothed coefficients.

    Parameters:
    SX : ndarray (M, d)
        Regressor matrix (M = N * T).
    SY : ndarray (M,) or (M, 1)
        Dependent variable vector.
    vt : ndarray (T,)
        Scaled time points (e.g., np.linspace(1/T, 1, T)).
    h : float
        Bandwidth for kernel smoothing.

    Returns:
    modY : ndarray (M,)
        Dependent variable with fixed effects removed.
    """
    T = len(vt)
    M, d = SX.shape
    N = M // T
    SY = SY.reshape(-1, 1)  # Ensure SY is column vector

    if d == 1:
        # Only intercept term; remove mean per individual
        SY_reshaped = SY.reshape(T, N, order='F')  # (T, N)
        means = np.mean(SY_reshaped, axis=0).reshape(-1,1)  # (N,)
        fixed_effects = np.kron(means, np.ones((T, 1)))  # (M, 1)
        modY = SY - fixed_effects

    else:
        Ebeta=find_coef(X_list,Y_panel, V, h,d)


        # Residuals after removing time-varying part for regressors 1 and beyond
        fitted = np.sum(SX[:, 1:] * Ebeta[:, 1:], axis=1, keepdims=True)  # (M, 1)
        residuals = SY - fitted

        # Remove individual means of residuals
        residuals_reshaped = residuals.reshape(T, N, order='F')  # (T, N)
        mean_residuals = np.mean(residuals_reshaped, axis=0).reshape(-1, 1) # (N,)
        fixed_effects = np.kron(mean_residuals, np.ones((T, 1)))  # (M, 1)
        modY = SY - fixed_effects


    return modY

#define a fct that creates a test set and a train test
def eval_func(MX_train_list, MY_train, MV_train, MX_test_list, MY_test, MV_test,h):
    """
       Eval function for kernel smoothing with cv_ksmooth that expects one u per country.

       Parameters:
           MX_train_list: list of (T_train, N) regressors
           MV_train: (T_train, N) index variable
           MY_train: (T_train, N) dependent variable
           MX_test_list: list of (T_test, N) regressors
           MV_test: (T_test, N) index variable
           MY_test: (T_test, N) dependent variable
           h: bandwidth scalar

       Returns:
           sum of squared errors (scalar)
       """
    t_test,N=MY_test.shape
    errors=[]
    for t in range(t_test):
        fitted = np.sum(
            np.vstack([A[t, :] for A in MX_test_list]) *
            cv_ksmooth(MX_train_list, MV_train, MY_train, MV_test[t], h),axis=0
        )
        errors.append(np.sum((MY_test[t] - fitted) ** 2))
    return errors

def panel_kfold_cv(MX_list, MY, MV,h,K, shuffle=False, random_state=None):
    """
        K-fold CV for panel data.

        X_list: list of arrays, each (T, N)
        Y: array (T, N)
        K: number of folds (split along time dimension T)
        eval_func: function(X_train_list, Y_train, X_test_list, Y_test) -> score
        shuffle: whether to shuffle time indices before folding

        Returns:
            mean_score, std_score, all_scores (array)
        """
    T,N=MY.shape
    if K>T:
        raise ValueError(f"Number of folds K={K} cannot be grater than number of points T={T}")
    time_indices=np.arange(T)
    if shuffle:
        rng=np.random.default_rng(random_state)
        rng.shuffle(time_indices)

    fold_sizes=np.full(K,T//K)
    fold_sizes[:T%K]+=1

    current=0
    results=[]
    for fold_size in fold_sizes:
        test_idx=time_indices[current:current+fold_size]
        train_idx=np.setdiff1d(time_indices, test_idx)
        current+=fold_size

        MX_train_list=[X[train_idx,:] for X in MX_list]
        MX_test_list=[X[test_idx,:] for X in MX_list]

        MY_train=MY[train_idx,:]
        MY_test=MY[test_idx, :]

        MV_train=MV[train_idx, :]
        MV_test=MV[test_idx,:]

        sum_errors_resid_squared_vector=eval_func(MX_train_list, MY_train, MV_train, MX_test_list, MY_test, MV_test,h)
        results.append(np.sum(sum_errors_resid_squared_vector))
    results=np.array(results)
    mean_score=results.mean()
    std_score=results.std(ddof=1)

    return mean_score, std_score, results




def hselector(X_list,Y_panel,V,SX,SY,vt,k_fold,h_len=10,remove_fixed_effects_variable=True):
    """
        Bandwidth selection for heterogeneous varying-coefficient panel data model using leave-one-out CV.

        Parameters:
        - SX: (M, d) array of explanatory variables (M = N * T)
        - SY: (M,) array of dependent variable
        - vt: (T,) array of scaled time (1/T, 2/T, ..., T/T)
        - h_len: number of bandwidth values to try

        Returns:
        - hopt: optimal bandwidth minimizing CVMSE
        """
    T=len(vt)
    M,d=SX.shape
    N=M//T

    # h is an array of h_len values, linearly spaced between 0.08*T^(-1/5) and 0.5*T^(-1/5)
    h = np.linspace(0.05 * (T ** (-1 / 5)), 0.5 * (T ** (-1 / 5)), 10)

    # CVMSE is a 1D numpy array filled with zeros, of length h_len
    CVMSE = np.zeros(h_len)
    CVSE=[]
    for k in range(h_len):
        if remove_fixed_effects_variable:
             SY_mod=remove_fixed_effects(X_list,Y_panel,V,SX,SY,vt,h[k])
        else:
            SY_mod=SY
        MY = SY_mod.reshape(T, N, order='F')  # Assuming Fortran order panel stacking
        size_fold=T//k_fold

        _, _, fold_sum_squared_errors = panel_kfold_cv(X_list, MY, V,h[k], k_fold)
        CVSE.append(fold_sum_squared_errors)
        CVMSE[k] = np.sum(fold_sum_squared_errors)/(M*size_fold)  # Use mean per-bandwidth+need the size of each fold
    CV_idx = np.argmin(CVMSE)
    hopt=h[CV_idx]
    return MY,CVSE,hopt


def weight_function(v, h):
    """
    Binary weight function that returns 1 where h <= v <= 1 - h, else 0.

    Parameters:
    - v: array-like or scalar values (usually scaled time index)
    - h: bandwidth (scalar)

    Returns:
    - res: boolean array or scalar mask
    """
    v = np.asarray(v)
    return (v >= h) & (v <= 1 - h)




def dist_func(SX, SY, Ebeta, h, vt):
    M, d = SX.shape  # M: number of samples, d: feature dim
    T = vt.shape[0]  # Number of time steps
    N = SY.shape[0] // T  # Number of sequences

    weights = weight_function(vt, h)  # Should return array of shape (T,)
    assert weights.shape[0] == T, "Weight function must return T weights"

    D = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            dist = 0.0
            for t in range(T):
                diff = Ebeta[i*T + t] - Ebeta[j*T + t]
                dist += np.linalg.norm(diff) * weights[t]
            D[i, j] = dist / T



    return D


def kcluster(N, K,D):
    """
    Cluster time-varying functional coefficients using hierarchical clustering.

    Parameters:
    - SX: (M, d) array of explanatory variables (M = N * T)
    - SY: (M,) array of dependent variable
    - Ebeta: (N, T*d) array of estimated functional coefficients
    - h: bandwidth for weight function
    - K: number of clusters
    - ksmooth: smoothing function (should follow same API as your `ksmooth`)
    - weight_function: binary weight function (e.g., returns mask for central time points)

    Returns:
    - Dictionary with:
        - 'mship': cluster memberships (N,)
        - 'EG': list of arrays of indices per group
        - 'EG_len': array of group sizes
        - 'Ealpha': (K, T*d) array of estimated average coefficient functions per group
    """




    """EKs = Kselector(SX, SY, Ebeta, h, Kmin, Kmax)
    EBICK = EKs[0]  # MATLAB indices start at 1, Python at 0
    EAICK = EKs[1]

    # Cluster into EBICK clusters
    cluster_labels = fcluster(Z, t=EBICK, criterion='maxclust')  # equivalent to MATLAB's cluster(Z, 'maxclust', EBICK)

    # Prepare groupings and arrays
    BICEG = [[] for _ in range(EBICK)]
    BICEG_len = np.zeros(EBICK, dtype=int)
    BICEalpha = np.zeros((EBICK, T * d))  # T and d must be defined earlier

    # Optional: populate BICEG and BICEG_len
    for idx, group in enumerate(cluster_labels):
        BICEG[group - 1].append(idx)
        BICEG_len[group - 1] += 1"""



    D_condensed=squareform(D)
    Z = linkage(D_condensed, method='complete')  # 'complete' linkage clustering
    mship = fcluster(Z, K, criterion='maxclust')  # Assign groups


    return Z,mship

#here I will get the ame cuntries but because python indexes differently it will be one value smaller instead of 1 it will be 0, instead of 2  will be 1 and so on so on
def membership (SX,SY,vt,mship,K,h):
    T=vt.shape[0]
    M=SY.shape[0]
    N=M//T
    d=SX.shape[1]


    EG = []
    EG_len = np.zeros(K, dtype=int)
    Ealpha = np.zeros((K, T * d))
    Ealpha_inv=np.zeros((K,T*d))

    for k in range(1, K + 1):
        members = np.where(mship == k)[0]
        EG.append(members)
        EG_len[k - 1] = len(members)
    for k in range(1, K + 1):
        members = np.where(mship == k)[0]
        # Get indices in full SX/SY corresponding to this group
        indices = np.vstack([
            np.arange(i * T, (i + 1) * T) for i in members
        ]).ravel()


        alpha_k = np.zeros((d, T))
        vt_group = np.tile(vt, (EG_len[k - 1], 1))  # shape: (T * k, 1)

        for t in range(T):
            u_eval = (t + 1) / T
            alpha_k[:, t] = ksmooth(SX[indices], vt_group.flatten(), SY[indices], u_eval, h).flatten()

        Ealpha[k - 1, :] = alpha_k.flatten()#used for display
        Ealpha_inv[k-1,:]=alpha_k.flatten(order='F')#used for computing NPMSE
    return {
        'mship': mship,
        'EG': EG,
        'EG_len': EG_len,
        'Ealpha': Ealpha,
        'Ealpha_inv':Ealpha_inv
    }


def compute_np_mse(SX, reshaped_Ebeta, SY_removed_effects, parts_filename="parts", row_sums_filename="row_sums"):
    """
    Compute NP MSE from given matrices, save intermediate results to files, and print the result.

    Parameters:
    - SX: np.ndarray, input matrix of shape (T*N, 3)
    - reshaped_Ebeta: np.ndarray, same shape as SX
    - SY_removed_effects: np.ndarray, shape (T*N, 1) or broadcastable to row_sums
    - parts_filename: str, file name to save parts matrix
    - row_sums_filename: str, file name to save row sums
    """

    # Element-wise multiplication and row-wise sum
    parts = SX * reshaped_Ebeta
    row_sums = np.sum(parts, axis=1).reshape(-1, 1)

    with open(parts_filename, "w") as f:
        f.write("Coefficients evaluated at each u in N by 3T dimension:\n")
        f.write(np.array2string(parts, threshold=np.inf))

    with open(row_sums_filename, "w") as f:
        f.write("Coefficients evaluated at each u in N by 3T dimension:\n")
        f.write(np.array2string(row_sums, threshold=np.inf))

    # Compute squared error and mean
    NPMSE = np.mean((SY_removed_effects - row_sums) ** 2)

    # Print result
    print(f'pre-clustering NP MSE: {NPMSE:.6f}\n')

    return NPMSE

def find_ebeta_groups(Ealpha,mship,N,T,d):
    Ebeta = np.zeros((N * T, d))
    for j in range(N):
          Ebeta[j*T:(j+1)*T,:]=Ealpha[mship[j]-1,:].reshape(T,d)
    return Ebeta

def variation(X_list, Y_panel_removed_effects, VT,SX,SY_removed_effects,vt,hopt,k):
    T = vt.shape[0]
    M = SY_removed_effects.shape[0]
    N = M // T
    d = SX.shape[1]
    Ebeta=find_coef(X_list, Y_panel_removed_effects, VT, hopt, d)
    D=dist_func(SX, SY_removed_effects, Ebeta, hopt, vt)
    Z,mship=kcluster(N, k,D)
    result=membership(SX, SY_removed_effects, vt, mship, k, hopt)
    Ealpha = result['Ealpha']
    mship = result['mship']
    EG = result['EG']
    EG_len = result['EG_len']
    Ealpha_inv = result['Ealpha_inv']

    Ebeta_groups=find_ebeta_groups(Ealpha_inv,mship,N,T,d)

    # Element-wise multiplication
    elementwise_product = SX * Ebeta_groups  # shape (M, d)

    # Sum across columns (i.e., per row)
    summed_contributions = np.sum(elementwise_product, axis=1).reshape(-1,1)  # shape (M,)

    SY_removed_effects=SY_removed_effects.reshape(-1,1)

    # Residuals

    residuals = SY_removed_effects - summed_contributions  # shape (M,)

    # Squared residuals
    squared_residuals = residuals ** 2  # shape (M,)
    big_ones = np.ones((N, 1))  # Shape should be (N, d)
    weights = weight_function(vt, hopt)


    new_weights = np.kron(big_ones, weights)  # Shape (M, 1)
    weighted_squared_residuals=squared_residuals*new_weights
    total_sum = np.sum(weighted_squared_residuals)  # sums all elements
    variation = total_sum / (N * T)
    min_NK = min(EG_len)
    AICrho_tuning = 2 / (min_NK * T * hopt)  # for AIC
    BICrho_tuning = np.log(min_NK * T * hopt) / (min_NK * T * hopt)  # for BIC
    return min_NK,AICrho_tuning,BICrho_tuning,variation

def Kselector(X_list, Y_panel_removed_effects, VT,SX,SY_removed_effects,vt,hopt,Kmin=1,Kmax=10):
    T = vt.shape[0]
    M = SY_removed_effects.shape[0]
    N = M // T
    d = SX.shape[1]

    size=Kmax+1-Kmin
    AICK = np.zeros((size,1))
    BICK=np.zeros((size,1))
    for k in range (Kmin, Kmax+1):
         min_NK,AICrho_tuning,BICrho_tuning,V2K=variation(X_list, Y_panel_removed_effects, VT,SX,SY_removed_effects,vt,hopt,k)
         AICK[k - Kmin ] = np.log(V2K) + k * AICrho_tuning
         BICK[k - Kmin] = np.log(V2K) + k * BICrho_tuning
    BICidxK = np.argmin(BICK)
    AICidxK = np.argmin(AICK)

    BICKhat = Kmin + BICidxK
    AICKhat = Kmin + AICidxK

    Khat = [BICKhat, AICKhat]

    return AICK,BICK,Khat



def accuracy(SX, SY,Ebeta,Ebeta_0, h, K,K1, mship0, Ealpha_0, vt,d):
    """
    Calculates the purity, NMI, and RMSE of HAC results.

    Parameters:
    - SX: (M x d) data matrix of explanatory variables
    - SY: (M,) data vector for the dependent variable
    - Ebeta: (N x Td) matrix of estimated functional coefficients
    - h: bandwidth
    - K: estimated number of groups
    - mship0: (N,) true group membership
    - alpha0: (K0 x T) true group-specific functional coefficients
    - dist_func: function for computing distances
    - kcluster: clustering function
    - membership: function to extract memberships and coefficients

    Returns:
    - dict with keys 'Purity', 'NMI', and 'RMSE'
    """
    N = mship0.shape[0]
    T = SY.shape[0] // N

    K0 = Ealpha_0.shape[0]
    EG_0 = [np.where(mship0 == k)[0] for k in range(1, K0 + 1)]
    EG_len_0 = np.array([len(g) for g in EG_0])

    #1.Compute RMSE without grouping just use Ebeta0 where everything is generated from groups so we consider already certain coefficients being the same along entities if from the same group then compute the Ebeta given the SY and SX from the DGP assuming no grouping structure just using the cv_ksmoothing function then compute this RMSE bwtween the coefficents

    # Reshape to (N, T, d) and then flatten to (N, T*d)
    Ebeta_hat = Ebeta.reshape(N, T, d)
    Ebeta_hat = Ebeta_hat.reshape(N, -1, order='F')

    # RMSE
    RMSE_NP = np.sum(np.sqrt(np.sum((Ebeta_hat - Ebeta_0) ** 2, axis=1) / T))/N
    # 2.Compute Oracle RMSE:
    # Assume group assignments are known (oracle setting),
    # and use our algorithm to estimate the coefficients.
    # Then, calculate the RMSE using the true coefficients from the data-generating process (DGP).

    result = membership(SX,SY, vt, mship0, K, h)
    Ealpha_oracle = result['Ealpha']

    # Oracle RMSE: average RMSE across N groups, where Ebeta_oracle and Ebeta0 are (N, T*d)
    diff = Ealpha_oracle[mship0 - 1, :] - Ealpha_0[mship0 - 1, :] # shape (N, T*d)
    RMSE_Orl_r = np.sum(np.sqrt(np.sum(diff ** 2, axis=1) / T)) / N

    # 3. Compute RMSE under estimated group membership:
    # Instead of assuming known group assignments (as in the Oracle case), we now estimate group memberships using our clustering function.
    # We then use the group-level coefficients Ealpha (of shape G x T*N), where G is the number of groups.
    # Unlike Ebeta (which repeats the same coefficients for each entity), Ealpha contains one set of coefficients per group.
    # We assign each entity to a group, retrieve the corresponding coefficients from Ealpha, and compute RMSE as before.
    # Note: We assume the number of groups (G) is known and matches the true number used in the data-generating process.
    D = dist_func(SX, SY, Ebeta, h, vt)
    Z,mship = kcluster(N, K1, D)
    result = membership(SX, SY, vt, mship, K1, h)
    Ealpha = result['Ealpha']
    mship = result['mship']
    EG = result['EG']
    EG_len = result['EG_len']

    # RMSE
    diff = Ealpha[mship - 1, :] - Ealpha_0[mship0 - 1, :]
    RMSE = np.sum(np.sqrt(np.sum(diff ** 2, axis=1) / T)) / N

    # 4. Compute clustering evaluation metrics:
    #    - First, calculate Entropy and Mutual Information between the true and predicted cluster assignments.
    #    - Then, compute Purity to measure the extent to which clusters contain a single class.
    #    - Finally, calculate Normalized Mutual Information (NMI) as a normalized measure of shared information between the clusterings.

    EG_probs = EG_len / N
    entropy_estimation = -np.sum(EG_probs * np.log2(EG_probs + 1e-12))

    EG_probs_0 = EG_len_0 / N
    entropy_true = -np.sum(EG_probs_0 * np.log2(EG_probs_0 + 1e-12))

    purity = np.zeros(K1)
    mutual_information = 0

    for k in range(K1):
        ak = np.zeros(K)
        for k0 in range(K):
            inter = len(set(EG[k]).intersection(EG_0[k0]))
            ak[k0] = inter
            if inter > 0:
                mutual_information += (inter / N) * np.log2((N * inter) / (len(EG[k]) * len(EG_0[k0])) + 1e-12)
        purity[k] = np.max(ak)

    purity_criterion = np.sum(purity) / N
    normalized_mutual_information = mutual_information / ((entropy_estimation + entropy_true) / 2)

    return {'RMSE_NP':RMSE_NP,'RMSE_Orl_r':RMSE_Orl_r,'RMSE': RMSE, 'Purity': purity_criterion, 'NMI': normalized_mutual_information}

#Tuekey's biweight kernel function
def G(u, miu, v):
    z = (u - miu) / v
    indicator = np.abs(z) <= 1
    return indicator * (1 - z**2)**2
#logistic function
def F(v, u, s):
    return 1 / (1 + np.exp(-(v - u) / s))



def generate_betas_logistic_group(gct, u):
    """
    Generate beta_0 and beta_1 for a given group (gct) and scalar u using F(u, center, scale),
    following the MATLAB convention:
        - First index 1 = beta_0
        - First index 2 = beta_1
        - Second index = group classification type (1, 2, 3)
    """
    if gct == 1:
        beta_0 = 6 * F(u, 0.5, 0.1)  # a11
        beta_1 = 3 * (2*u - 4*u**2 + 2*u**3 + F(u, 0.6, 0.1))  # a21

    elif gct == 2:
        beta_0 = 6 * (2*u - 6*u**2 + 4*u**3 + F(u, 0.7, 0.05))  # a12
        beta_1 = 3 * (u - 3*u**2 + 2*u**3 + F(u, 0.7, 0.04))    # a22

    elif gct == 3:
        beta_0 = 6 * (4*u - 8*u**2 + 4*u**3 + F(u, 0.6, 0.05))  # a13
        beta_1 = 3 * (0.5*u - 0.5*u**2 + F(u, 0.4, 0.07))       # a23

    else:
        raise ValueError(f"Invalid group index: {gct}")

    return (beta_0, beta_1)



def compute_beta0(mship0, vt, generate_betas_func, normalize_first_beta=True):
    """
    Generalized version of compute_beta0.

    Parameters:
    - mship0: (N,) array of group memberships (1-based indexing)
    - vt: (T,) array of time values
    - generate_betas_func: function (gct, u) -> tuple of beta values
    - normalize_first_beta: whether to normalize the first beta across time per group

    Returns:
    - beta0: (N, num_betas * T) matrix of time-varying coefficients
    """
    N = len(mship0)
    T = len(vt)
    num_groups = len(np.unique(mship0))

    # Get number of betas dynamically from first call
    test_betas = generate_betas_func(1, vt[0])
    num_betas = len(test_betas)

    # Initialize alpha arrays: one per beta, shape (num_groups, T)
    alpha = np.zeros((num_betas, num_groups, T))

    # Compute group-level time curves
    for gct in range(1, num_groups + 1):
        beta_group = [[] for _ in range(num_betas)]
        for u in vt:
            betas = generate_betas_func(gct, u)
            for b in range(num_betas):
                beta_group[b].append(betas[b])
        for b in range(num_betas):
            beta_array = np.array(beta_group[b])
            if b == 0 and normalize_first_beta:
                # Optionally normalize first beta
                beta_array = 0.5 * (beta_array - np.mean(beta_array))
            alpha[b, gct - 1, :] = beta_array

    # Assign to individuals
    beta0 = np.zeros((N, num_betas * T))
    for i in range(N):
        g = mship0[i] - 1  # 0-based index
        for b in range(num_betas):
            beta0[i, b * T:(b + 1) * T] = alpha[b, g, :]
    # Compute group-level alpha0 (num_groups, T * num_betas)
    alpha0 = np.zeros((num_groups, num_betas * T))
    for g in range(num_groups):
        for b in range(num_betas):
            alpha0[g, b * T:(b + 1) * T] = alpha[b, g, :]

    return alpha0,beta0


def generate_alphas_logistic(i):
    alpha = np.random.normal(loc=0, scale=1)
    return alpha
def generate_betas_biweight(gct, u):
    """
    Generate beta_0 using a mixture of kernels based on group category (gct) and input scalar u.
    Returns beta_0 and a normally-distributed error.
    """
    beta_0 = 0

    if gct == 1:
        beta_0 = G(u, 1/2, 1/2)

    elif gct == 2:
        beta_0 = G(u, 1/4, 1/4) + G(u, 3/4, 1/4)

    elif gct == 3:
        beta_0 = G(u, 1/8, 1/8) + G(u, 3/8, 1/8) + G(u, 3/4, 1/4)

    elif gct == 4:
        beta_0 = G(u, 1/4, 1/4) + G(u, 5/8, 1/8) + G(u, 7/8, 1/8)

    elif gct == 5:
        beta_0 = G(u, 1/12, 1/12) + G(u, 1/4, 1/12) + G(u, 5/12, 1/12) + G(u, 3/4, 1/4)

    elif gct == 6:
        beta_0 = G(u, 1/4, 1/4) + G(u, 7/12, 1/12) + G(u, 3/4, 1/12) + G(u, 11/12, 1/12)

    else:
        raise ValueError(f"Invalid group index: {gct}")

    return (beta_0,)


def mship_biweight(N):
    if N % 6 != 0:
        raise ValueError("N must be divisible by 6")
    block_size = N // 6
    mship=np.repeat(np.arange(1, 7), block_size)
    return mship


def mship_logistic(N):
    N1 = int(0.3 * N)
    N2 = int(0.3 * N)
    N3 = N - N1 - N2  # ensure total length is exactly N
    return np.concatenate([
        np.full(N1, 1),
        np.full(N2, 2),
        np.full(N3, 3)
    ])

def monte_carlo_bandwidth(beta0, N, T, vt, reps=5, include_gamma=True,apply_fixed_effects=True,st_dev=1):
    M = N * T
    d = beta0.shape[1] // T  # Number of regressors

    sum_hopt = 0

    # Extract beta coefficients for each regressor
    beta_list = [
        beta0[:, i * T:(i + 1) * T]  # Shape (N, T)
        for i in range(d)
    ]

    for r in range(reps):

        # Generate idiosyncratic errors
        Se = np.random.normal(0, st_dev, M)
        # Optionally generate individual effects
        if include_gamma:
            gamma0 = np.random.normal(0, 1, N)
            gamma_repeat = np.kron(gamma0, np.ones(T))
        else:
            gamma_repeat = np.zeros(M)



        # Generate regressors
        SX_list = []
        for k in range(d):
            if k == 0:
                SX_list.append(np.ones(M))  # Intercept
            else:
                SX_list.append(np.random.normal(0, 1, M))

        # Compute linear component using beta and regressors
        linear_part = np.zeros(M)
        for i in range(d):
            beta_flat = beta_list[i].T.reshape(M)
            linear_part += SX_list[i] * beta_flat

        # Generate outcome
        SY = gamma_repeat + linear_part + Se

        # Reshape for panel format
        Y_panel = SY.reshape((T, N), order='F')
        VT = np.tile((np.arange(1, T + 1) / T).reshape(-1, 1), N)
        X_list = [sx.reshape((T, N), order='F') for sx in SX_list]
        SX = np.column_stack(SX_list)

        # Bandwidth selection
        _, _, hopt = hselector(X_list, Y_panel, VT, SX, SY, vt, T,h_len=10,remove_fixed_effects_variable=apply_fixed_effects)
        sum_hopt += hopt

    return sum_hopt / reps


def run_monte_carlo_np_estimation(mship0,Ealpha_0,Ebeta_0, N, T, vt, h,K, R=200, include_gamma=True, apply_fixed_effects=True,st_dev=1,k_min=4,k_max=8):
    M = N * T
    d = Ebeta_0.shape[1] // T  # Number of regressors
    # Initialize AIC and BIC evaluation vectors for R replications
    # Initialize frequency counters
    Klen = k_max - k_min + 1

    Freq_BICK = np.zeros(Klen, dtype=int)
    Freq_AICK = np.zeros(Klen, dtype=int)

    RMSE_NP_rep_AIC = np.zeros(R)
    Oracle_RMSE_rep_AIC = np.zeros(R)
    RMSE_rep_AIC = np.zeros(R)
    Purity_rep_AIC = np.zeros(R)
    NMI_rep_AIC = np.zeros(R)

    RMSE_NP_rep_BIC = np.zeros(R)
    Oracle_RMSE_rep_BIC = np.zeros(R)
    RMSE_rep_BIC = np.zeros(R)
    Purity_rep_BIC = np.zeros(R)
    NMI_rep_BIC = np.zeros(R)


    for r in range(R):

        # Generate error term
        Se = np.random.normal(0, st_dev, M)
        # Optionally generate country-specific fixed effects
        if include_gamma:
            gamma0 = np.random.normal(0, 1, N)
            gamma_repeat = np.kron(gamma0, np.ones(T))
        else:
            gamma_repeat = np.zeros(M)


        # Generate SX: list of d regressors
        SX_list = []
        for k in range(d):
            if k == 0:
                SX_list.append(np.ones(M))  # Intercept
            else:
                SX_list.append(np.random.normal(0, 1, M))

        # Stack into SX (M x d)
        SX = np.column_stack(SX_list)

        # Flatten beta coefficients for each regressor
        beta_flat_list = []
        for i in range(d):
            beta_i = Ebeta_0[:, i * T:(i + 1) * T]  # Shape (N, T)
            beta_flat_list.append(beta_i.reshape(M))

        # Generate outcome
        linear_part = sum(SX[:, i] * beta_flat_list[i] for i in range(d))
        SY = gamma_repeat + linear_part + Se

        # Reshape for panel format
        Y_panel = SY.reshape((T, N), order='F')
        VT = np.tile((np.arange(1, T + 1) / T).reshape(-1, 1), N)
        X_list = [sx.reshape((T, N), order='F') for sx in SX.T]

        if apply_fixed_effects:
            # Fixed effects removal
            SY_detrended = remove_fixed_effects(X_list, Y_panel, VT, SX, SY, vt, h)
            Y_panel_removed_effects = SY_detrended.reshape((T, N), order='F')
        else:
            SY_detrended=SY
            Y_panel_removed_effects=Y_panel

        # Nonparametric estimation
        Ebeta = find_coef(X_list, Y_panel_removed_effects, VT, h, d)
        vt = vt.reshape(-1, 1)

        AICK,BICK,[BICKhat, AICKhat]=Kselector(X_list,Y_panel_removed_effects,VT,SX,SY_detrended,vt,h,Kmin=k_min,Kmax=k_max)
        # Update frequencies

        K_range = np.arange(k_min, k_max + 1)  # includes Kmax

        Freq_BICK += (BICKhat == K_range)
        Freq_AICK += (AICKhat == K_range)



        result_BIC=accuracy(SX,SY_detrended,Ebeta,Ebeta_0,h,K,BICKhat,mship0,Ealpha_0,vt,d)
        result_AIC=accuracy(SX,SY_detrended,Ebeta,Ebeta_0,h,K,AICKhat,mship0,Ealpha_0,vt,d)


        RMSE_NP_rep_BIC[r] = result_BIC['RMSE_NP']
        Oracle_RMSE_rep_BIC[r] = result_BIC['RMSE_Orl_r']
        RMSE_rep_BIC[r] =  result_BIC['RMSE']
        Purity_rep_BIC[r] = result_BIC['Purity']
        NMI_rep_BIC[r] = result_BIC['NMI']

        RMSE_NP_rep_AIC[r] = result_AIC['RMSE_NP']
        Oracle_RMSE_rep_AIC[r] = result_AIC['RMSE_Orl_r']
        RMSE_rep_AIC[r] = result_AIC['RMSE']
        Purity_rep_AIC[r] = result_AIC['Purity']
        NMI_rep_AIC[r] = result_AIC['NMI']

    return {

        'AIC': {
            'Freq':Freq_AICK,
            'RMSE_NP': RMSE_NP_rep_AIC,
            'Oracle_RMSE': Oracle_RMSE_rep_AIC,
            'RMSE': RMSE_rep_AIC,
            'Purity': Purity_rep_AIC,
            'NMI': NMI_rep_AIC,
        },
        'BIC': {
            'Freq': Freq_BICK,
            'RMSE_NP': RMSE_NP_rep_BIC,
            'Oracle_RMSE': Oracle_RMSE_rep_BIC,
            'RMSE': RMSE_rep_BIC,
            'Purity': Purity_rep_BIC,
            'NMI': NMI_rep_BIC,
        }
    }


def compute_summary_stats(NMI_rep_BIC, NMI_rep_AIC, Purity_rep_BIC, Purity_rep_AIC,
                          Oracle_RMSE_rep_BIC, RMSE_NP_rep_BIC,
                          RMSE_rep_BIC, RMSE_rep_AIC):
    stats = {
        'ave_BICNMI': np.mean(NMI_rep_BIC), 'std_BICNMI': np.std(NMI_rep_BIC),
        'ave_BICPurty': np.mean(Purity_rep_BIC), 'std_BICPurty': np.std(Purity_rep_BIC),
        'ave_AICNMI': np.mean(NMI_rep_AIC), 'std_AICNMI': np.std(NMI_rep_AIC),
        'ave_AICPurty': np.mean(Purity_rep_AIC), 'std_AICPurty': np.std(Purity_rep_AIC),
        'ave_RMSE_Orl': np.mean(Oracle_RMSE_rep_BIC), 'std_RMSE_Orl': np.std(Oracle_RMSE_rep_BIC),
        'ave_RMSE_NP': np.mean(RMSE_NP_rep_BIC), 'std_RMSE_NP': np.std(RMSE_NP_rep_BIC),
        'ave_BICRMSE_HAC': np.mean(RMSE_rep_BIC), 'std_BICRMSE_HAC': np.std(RMSE_rep_BIC),
        'ave_AICRMSE_HAC': np.mean(RMSE_rep_AIC), 'std_AICRMSE_HAC': np.std(RMSE_rep_AIC),
    }
    return stats


import matplotlib.pyplot as plt
import numpy as np


def plot_group_coefficients(YV, Ealpha, T, num_groups, num_regresors=3, xlim=(1970, 2020), ylim_list=None):
    """
    Plots estimated group-specific functional coefficients in a grid.

    Parameters:
    - YV: array-like, x-axis values
    - BICEalpha: 2D numpy array or similar, shape (num_groups, T*num_subgroups)
    - T: int, length of each subgroup
    - num_groups: int, number of groups (default 4)
    - num_subgroups: int, number of subgroups (default 3)
    - xlim: tuple, x-axis limits
    - ylim_list: list of tuples, y-axis limits for each subplot (optional)
    """
    plt.figure(figsize=(15, 8))
    total_plots = num_groups * num_regresors

    # default ylims if none provided
    if ylim_list is None:
        ylim_list = [(None, None)] * total_plots

    for sg in range(num_regresors):
        for g in range(num_groups):
            idx = sg * num_groups + g
            plt.subplot(num_regresors, num_groups, idx + 1)

            start_col = sg * T
            end_col = (sg + 1) * T

            plt.plot(YV, Ealpha[g, start_col:end_col])
            plt.xlim(xlim)
            if ylim_list[idx] != (None, None):
                plt.ylim(ylim_list[idx])

            plt.title(f"$\\hat{{\\gamma}}_{{{g + 1},{sg}}}(u)$", fontsize=10)
            plt.xlabel('u')
            plt.tight_layout()
    plt.show()


def plot_grouped_time_series(YV, YX_list, EG,num_groups, titles, xlim=(1970, 2020), num_regressors_dependent=3):
    """
    Plots grouped time series data with subplots.


    Parameters:
    - YV: x-axis values (time)
    - data_groups: list of 2D arrays (time x samples), e.g. [Y, X2, X3]
    - group_indices: list of lists with indices of samples per group, e.g. BICEG from your code
    - titles: list of titles for each data group, e.g. ['GY', 'GK', 'GPOP']
    - xlim: tuple for x-axis limits
    - nrows: number of subplot rows
    - ncols: number of subplot cols
    """
    plt.figure(figsize=(18, 12))


    for i, data in enumerate(YX_list):
        for g in range(num_groups):
            plt.subplot(num_regressors_dependent, num_groups, i * num_groups + g + 1)
            plt.xlim(xlim)

            for idx in EG[g]:
                plt.plot(YV, data[:, idx])
            plt.title(f"{titles[i]} -- Group {g + 1}")
            plt.tight_layout()
    plt.show()

