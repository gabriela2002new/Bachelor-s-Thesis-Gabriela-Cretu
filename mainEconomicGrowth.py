from collections import Counter
import pandas as pd
from matplotlib import pyplot as plt
import FunctionsExtensions
import FunctionsLatentTimeVarying  # your helper module
from FunctionsLatentTimeVarying import remove_fixed_effects, find_ebeta_groups
import scipy.stats as stats
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az


def main3():
    #Part 1
    #here we first select the variables from the data frame then move from wide format to long format using dufferet functions and then the using another function we select the actual varaibels
    #in the sought after format and finally we also consider this vector with the normalized times to find the smooyhing functions
    #over time following to plot each variable for each country with the sole purpose to see how their growth indicators fluctuate
    #also define the pooled ones where instead ofa panel format we conder vectors NT*1(vectorization of what we had before in the long format for each variable)
    file_path = r'C:\Users\acer\PycharmProjects\LatentGroupsWithTimeVaryingCoefficients\WDI_EconomicGrowth (3).xlsx'
    year_columns=list(range(1971, 2017))
    df=FunctionsLatentTimeVarying.load_year_data(file_path, 1971, 2017)

    # Step 1: Load and reshape the panel data
    print(df)
    # Step 2: Prepare X and Y matrices for regression
    df =FunctionsLatentTimeVarying.load_panel_data_pov_gap(df, year_columns)
    with open("transformed_panel_data.txt", "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))

    # Get unique countries in the order they appear
    unique_countries = df['Country'].drop_duplicates()

    # Print or save to text file
    with open("countries.txt", "w", encoding="utf-8") as f:
        for country in unique_countries:
            f.write(f"{country}\n")

    regressors_list = ['Population growth', 'Capital formation']
    y_var = 'GDP growth'
    intercept='Country mean (intercept)'
    x_limit=(1970, 2020)

    Y_panel, X_list, SX, SY, YV, vt, VT, u, h, N, T, pivoted, d = FunctionsExtensions.prepare_panel_data_with_dynamic_names_unpacked(
        df,
        y_var,
        regressors_list,
        start_year=1971,
        end_year=2016,
        h=0.1, d=3
    )

    print(SX.shape)
    print(YV.shape)
    print(X_list[0].shape)

    # Plotting
    var_names = [y_var] + regressors_list
    FunctionsExtensions.plot_panel_variables(YV, Y_panel, X_list, var_names, xlim=x_limit, ncols=d)


    # Step 5: Apply kernel smoothing
    coef_estimates = FunctionsLatentTimeVarying.cv_ksmooth(X_list, VT, Y_panel, u, h)

    # Step 6: Label and display results
    row_labels = [intercept]+regressors_list
    country_names = pivoted[y_var].columns.tolist()
    coef_df = pd.DataFrame(coef_estimates, index=row_labels, columns=country_names)

    print("Estimated coefficients (rows = predictors, columns = countries):")

    with open("CVKsmooth_specific.txt", "w") as f:
        f.write("Estimated coefficients (rows = predictors, columns = countries):\n")
        f.write(coef_df.to_string())  # Convert DataFrame to string for writing

    SX = np.asarray(SX, dtype=np.float64)
    SY = np.asarray(SY, dtype=np.float64)
    vt = np.asarray(vt, dtype=np.float64)

    coef=FunctionsLatentTimeVarying.find_coef(X_list, Y_panel, VT, h,d)
    with open("coefficients_overall.txt","w") as f:
        f.write("Coefficients evaluated at each u :\n")
        f.write(np.array2string(coef,threshold=np.inf))

    result = FunctionsLatentTimeVarying.remove_fixed_effects(X_list,Y_panel,VT,SX,SY, vt, h)
    result_T_by_N = result.reshape(T, N)  # Replace T and N with actual values
    with open("removed_fixed_effects.txt", "w") as f:
        f.write("Fixed effect-removed result:\n")
        f.write(np.array2string(result, threshold=np.inf))

    #select the best bandwidth
    k_fold=T
    MY,CVSE,hopt=FunctionsLatentTimeVarying.hselector(X_list, Y_panel, VT, SX, SY, vt, T, h_len=10)

    print(hopt)

    optimal_h_list = []
    """for t in range(T - 1):
        _,_,hopt = FunctionsLatentTimeVarying.hselector([X1, X2, X3], Y_panel, VT, SX, SY, vt, t + 2, h_len=10)
        optimal_h_list.append(hopt)  # convert to tuple here

    print(optimal_h_list)
    counts = Counter(optimal_h_list)
    most_common_h, freq = counts.most_common(1)[0]

    print(f"Bandwidth {most_common_h} is optimal in {freq} folds, selected as best overall.")

    # Print frequency of each distinct element
    for value, freq in counts.items():
        print(f"Value: {value}  - Frequency: {freq}")"""

    matrix = np.arange(N * T * d).reshape(N * T, d)
    print(matrix)
    SY_removed_effects=FunctionsLatentTimeVarying.remove_fixed_effects(X_list,Y_panel,VT,SX,SY,vt,hopt)
    with open("removed_effects_depend.txt","w") as f:
        f.write(" evaluated at each u in N by 3T dimension:\n")
        f.write(np.array2string(SY_removed_effects,threshold=np.inf))

    Y_panel_removed_effects = SY_removed_effects.reshape((T, N), order='F')  # 'F' for column-major order
    with open("panel_removed_effects_coefficients.txt","w") as f:
        f.write("Coefficients evaluated at each u in N by 3T dimension:\n")
        f.write(np.array2string(Y_panel_removed_effects,threshold=np.inf))

    coef = FunctionsLatentTimeVarying.find_coef(X_list, Y_panel_removed_effects, VT, hopt,d)


    # Now reshape to (N, 3T) by stacking the T rows horizontally
    Ebeta= coef.reshape( T * d,N,order='F').T
    with open("reshaped_coefficients.txt","w") as f:
        f.write("Coefficients evaluated at each u in N by 3T dimension:\n")
        f.write(np.array2string(Ebeta,threshold=np.inf))

    # Assuming SY, SX, Ebeta, M, and T are already defined numpy arrays or variables
    # Reshape Ebeta similar to the MATLAB indexing
    M=N*T


    # Stack the reshaped parts horizontally (i.e., columns)
    reshaped_Ebeta = coef
    with open("reshaped_Ebeta.txt", "w") as f:
        f.write("Coeffficients reshaped:\n")
        f.write(np.array2string(reshaped_Ebeta, threshold=np.inf))

    FunctionsLatentTimeVarying.compute_np_mse(SX, reshaped_Ebeta, SY_removed_effects, parts_filename="parts", row_sums_filename="row_sums")

    #Compute the distance between pairwise points
    D=FunctionsLatentTimeVarying.dist_func(SX, SY,reshaped_Ebeta,hopt,vt)
    print(D)

    with open("distances_ebeta", "w") as f:
        f.write("Pairwise distances between coefficients:\n")
        for row in D:
            f.write(" ".join(str(value) for value in row) + "\n")

    """with open("distances_ebeta", "w") as f:
        f.write("Condensed pairwise distances between coefficients:\n")
        f.write(" ".join(str(value) for value in distances) + "\n")"""""

    K_clusters=4
    Z,mship=FunctionsLatentTimeVarying.kcluster(N,K_clusters,D)

    result = FunctionsLatentTimeVarying.membership(SX,SY_removed_effects, vt,mship,K_clusters, hopt)
    Ealpha = result['Ealpha']
    mship=result['mship']
    EG = result['EG']
    EG_len = result['EG_len']
    Ealpha_inv=result['Ealpha_inv']

    with open('membership_result.txt', 'w') as f:
        for key, value in result.items():
            f.write(f"{key}:\n{value}\n\n")

    ebeta_groups=FunctionsLatentTimeVarying.find_ebeta_groups(Ealpha_inv,mship,N,T,d)
    with open("Ebeta_groups.txt", "w") as f:
        f.write("Coeffficients groups ebeta reshaped:\n")
        f.write(np.array2string(ebeta_groups, threshold=np.inf))
    FunctionsLatentTimeVarying.compute_np_mse(SX, ebeta_groups, SY_removed_effects, parts_filename="parts_groups",
                                              row_sums_filename="row_sums_groups")

    min_NK,AICrho_tuning,BICrho_tuning,variation=FunctionsLatentTimeVarying.variation(X_list, Y_panel_removed_effects, VT, SX, SY_removed_effects, vt, hopt, 4)

    with open("variation.txt", "w") as f:
        f.write(f"return {min_NK}, {AICrho_tuning}, {BICrho_tuning}, {variation}\n")

    AICK,BICK,[BICKhat, AICKhat]=FunctionsLatentTimeVarying.Kselector(X_list, Y_panel_removed_effects, VT, SX, SY_removed_effects, vt, hopt)
    with open("Kselector.txt", "w") as f:
        f.write(f"return {AICK}, {BICK}, {[BICKhat, AICKhat]}\n")
    Z_AIC, mship_AIC = FunctionsLatentTimeVarying.kcluster(N, AICKhat, D)
    Z_BIC, mship_BIC=FunctionsLatentTimeVarying.kcluster(N, BICKhat, D)

    result= FunctionsLatentTimeVarying.membership(SX, SY_removed_effects, vt, mship_AIC, AICKhat, hopt)
    AICEalpha = result['Ealpha']
    AICmship = result['mship']
    AICEG = result['EG']
    AICEG_len = result['EG_len']
    AICEalpha_inv = result['Ealpha_inv']

    result = FunctionsLatentTimeVarying.membership(SX, SY_removed_effects, vt, mship_BIC, BICKhat, hopt)
    BICEalpha = result['Ealpha']
    BICmship = result['mship']
    BICEG = result['EG']
    BICEG_len = result['EG_len']
    BICEalpha_inv = result['Ealpha_inv']

    with open('membership_result_economic_growth.txt', 'w') as f:
        for key, value in result.items():
            f.write(f"{key}:\n{value}\n\n")


    countries = unique_countries

    groups = BICEG
    # Create list of dicts to make DataFrame
    table_rows = []
    for i, group in enumerate(groups, start=1):
        group_countries = [countries.iloc[idx] for idx in group]

        table_rows.append({"Group": f"Group {i}", "Countries": ", ".join(group_countries)})

    df_countries = pd.DataFrame(table_rows)

    latex_table = df_countries.to_latex(index=False, column_format='|c|p{12cm}|', escape=False)
    # Save to a file
    with open("groups_countries_table.tex", "w") as f:
        f.write(latex_table)

    print("LaTeX table saved to groups_countries_table.tex")

    print(d)
    FunctionsLatentTimeVarying.plot_group_coefficients(YV, BICEalpha , T,BICKhat , num_regresors=d, xlim=x_limit, ylim_list=None)
    YX_list = [Y_panel] + X_list[1:]
    FunctionsLatentTimeVarying.plot_grouped_time_series(YV,YX_list,BICEG,BICKhat,var_names,xlim=x_limit, num_regressors_dependent=d)

    """G=4
    ebeta=coef
    unit_idx = np.repeat(np.arange(N), T)  # shape: (N*T,)
    time_idx = np.tile(np.arange(T), N)  # shape: (N*T,)




    np.random.seed(123)



    # Suppose you have the following data:
    # ebeta_ntd: shape (N, T, d) = smoothed coefficients per unit, over time, per variable



    with pm.Model() as model:
        # Group assignments (discrete)
        g = pm.Categorical("g", p=np.ones(G) / G, shape=N)  # One group per unit

        # Group-level beta trajectories (Random Walk over time)
        beta_group = pm.GaussianRandomWalk("beta_group", sigma=0.1, shape=(G, T, d))

        # Expand g to match (i,t)
        g_full = g[unit_idx]  # shape (N*T,)

        # Select coefficients for each (i,t)
        beta_it = beta_group[g_full, time_idx, :]  # shape (N*T, d)

        # Observation noise
        sigma_obs = pm.Exponential("sigma_obs", 1.0)

        # Likelihood
        obs = pm.Normal("obs", mu=beta_it, sigma=sigma_obs, observed=ebeta)

        # Sampling
        trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)

    # --- Extract results ---

    # Posterior mean of beta_group: shape (G_max, T, d)
    beta_group_samples = trace.posterior["beta_group"]  # xarray DataArray
    beta_group_mean = beta_group_samples.mean(dim=["chain", "draw"]).values

    # Posterior samples of group assignments g: shape (chains, draws, N*T)
    g_samples = trace.posterior["g"].stack(sample=("chain", "draw")).values  # shape (samples, N*T)

    # Compute mode (most frequent group) per (i,t)
    mode_groups = stats.mode(g_samples, axis=0).mode[0]
    mode_groups = mode_groups.reshape(N, T)

    # Construct estimated beta_it for each unit i and time t
    beta_it_estimated = np.zeros((N, T, d))
    for i in range(N):
        for t in range(T):
            grp = mode_groups[i, t]
            beta_it_estimated[i, t, :] = beta_group_mean[grp, t, :]

    print("Estimated beta coefficients shape:", beta_it_estimated.shape)
    print("Sample estimated beta coefficients for first unit, first time:", beta_it_estimated[0, 0, :])"""


if __name__ == '__main__':
    main3()

