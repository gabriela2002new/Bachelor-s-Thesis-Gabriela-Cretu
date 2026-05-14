from collections import Counter

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.ma.extras import row_stack

import FunctionsLatentTimeVarying  # your helper module
import FunctionsExtensions

x=10

def main():
    #Part 1
    #here we first select the variables from the data frame then move from wide format to long format using dufferet functions and then the using another function we select the actual varaibels
    #in the sought after format and finally we also consider this vector with the normalized times to find the smooyhing functions
    #over time following to plot each variable for each country with the sole purpose to see how their growth indicators fluctuate
    #also define the pooled ones where instead ofa panel format we conder vectors NT*1(vectorization of what we had before in the long format for each variable)


    file_path_Poverty_gap=r'C:\Users\acer\PycharmProjects\LatentGroupsWithTimeVaryingCoefficients\Data\Poverty_gap_new.xlsx'
    file_path_Political_corruption=r'C:\Users\acer\PycharmProjects\LatentGroupsWithTimeVaryingCoefficients\Data\Political_corruption.xlsx'
    file_path_Gini_coefficient=r'C:\Users\acer\PycharmProjects\LatentGroupsWithTimeVaryingCoefficients\Data\Gini_coefficient.xlsx'
    file_path_Human_Development_Index=r'C:\Users\acer\PycharmProjects\LatentGroupsWithTimeVaryingCoefficients\Data\Human_Development_Index.xlsx'
    file_path_Government_expenditure=r'C:\Users\acer\PycharmProjects\LatentGroupsWithTimeVaryingCoefficients\Data\Government_expenditure.xlsx'

    df_Political = FunctionsExtensions.load_data(file_path_Political_corruption)
    df_Political = FunctionsExtensions.filter_countries(df_Political )
    df_Political=FunctionsExtensions.filter_top_countries_by_data_count(df_Political ,country_col='Country',year_col='Year')


    df_Gini = FunctionsExtensions.load_data(file_path_Gini_coefficient)
    df_Gini = FunctionsExtensions.filter_countries(df_Gini)
    df_Gini = FunctionsExtensions.filter_top_countries_by_data_count(df_Gini, country_col='Country',year_col='Year')


    df_gov_expenditure = FunctionsExtensions.load_data(file_path_Government_expenditure)
    df_gov_expenditure = FunctionsExtensions.filter_countries(df_gov_expenditure)
    df_gov_expenditure = FunctionsExtensions.filter_top_countries_by_data_count(df_gov_expenditure, country_col='Country',year_col='Year')


    df_HDI = FunctionsExtensions.load_data(file_path_Human_Development_Index)
    df_HDI = FunctionsExtensions.filter_countries(df_HDI)
    df_HDI = FunctionsExtensions.filter_top_countries_by_data_count(df_HDI, country_col='Country',year_col='Year')


    df_poverty_gap = FunctionsExtensions.load_data(file_path_Poverty_gap)
    with open("Poverty_gap_data.txt", "w") as f:
            f.write("Interpolated poverty gap index values per country:\n\n")
            f.write(df_poverty_gap.to_string(index=False))
    df_poverty_gap = FunctionsExtensions.filter_countries(df_poverty_gap)
    df_poverty_gap = FunctionsExtensions.filter_top_countries_by_data_count(df_poverty_gap, country_col='Country',year_col='Year',top_n=200,min_y=1989)


    dfs=[df_Political,df_Gini,df_gov_expenditure,df_HDI,df_poverty_gap]

    [df_Political_filtered,df_Gini_filtered,df_gov_expenditure_filtered,df_HDI_filtered,df_poverty_gap_filtered]=FunctionsExtensions.filter_dfs_to_common_countries(dfs, country_col='Country')
    #Political corruption index

    dfs = [df_Political, df_Gini, df_gov_expenditure, df_HDI]

    [df_Political_filtered, df_Gini_filtered, df_gov_expenditure_filtered, df_HDI_filtered] = FunctionsExtensions.filter_dfs_to_common_countries(dfs, country_col='Country')


    df_interpolated_Political_corruption=FunctionsExtensions.interpolate_pchip(df_Political_filtered, value_col='Political_corruption', year_col='Year', country_col='Country',
                           return_start_year=1989)
    with open("Political_corruption_interpolated.txt", "w") as f:
        f.write("Interpolated Political Corruption Index values per country:\n\n")
        f.write(df_interpolated_Political_corruption.to_string(index=False))

    #Gini Coefficient Index

    df_interpolated_Gini_coefficient = FunctionsExtensions.interpolate_pchip(df_Gini_filtered, value_col='Gini_coefficient', year_col='Year',
                                                            country_col='Country',
                                                            return_start_year=1989)
    with open("Gini_coefficient_interpolated.txt", "w") as f:
        f.write("Interpolated Gini coefficient index values per country:\n\n")
        f.write(df_interpolated_Gini_coefficient.to_string(index=False))

    # Government expenditure

    df_interpolated_Government_expenditure = FunctionsExtensions.interpolate_pchip(df_gov_expenditure_filtered, value_col='Government_expenditure', year_col='Year',
                                                            country_col='Country',
                                                            return_start_year=1989)
    with open("Government_expenditure_interpolated.txt", "w") as f:
        f.write("Interpolated government expenditure values per country:\n\n")
        f.write(df_interpolated_Government_expenditure.to_string(index=False))

    # Human Development Index

    df_interpolated_Human_Development_Index = FunctionsExtensions.interpolate_pchip(df_HDI_filtered, value_col='Human_Development_Index', year_col='Year',
                                                            country_col='Country',
                                                            return_start_year=1989)
    with open("Human_Development_Index_interpolated.txt", "w") as f:
        f.write("Interpolated human development index values per country:\n\n")
        f.write(df_interpolated_Human_Development_Index.to_string(index=False))

    #Poverty gap

    df_interpolated_poverty_gap = FunctionsExtensions.interpolate_pchip(df_poverty_gap_filtered, value_col='Poverty_gap',
                                                                year_col='Year',
                                                                country_col='Country',
                                                                return_start_year=1989)
    with open("Poverty_gap_new_interpolated.txt", "w") as f:
            f.write("Interpolated poverty gap index values per country:\n\n")
            f.write(df_interpolated_poverty_gap.to_string(index=False))

    # Your dataframes, one per variable
    dfs = [df_interpolated_poverty_gap, df_interpolated_Human_Development_Index,df_interpolated_Political_corruption, df_interpolated_Gini_coefficient,df_interpolated_Government_expenditure]
    dfs1=[df_interpolated_Human_Development_Index,df_interpolated_Political_corruption, df_interpolated_Gini_coefficient,df_interpolated_Government_expenditure]

    # Combine them into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df = combined_df.sort_values(by=['Country','variable']).reset_index(drop=True)
    combined_df = combined_df[combined_df['Country'] != 'Paraguay']


    with open("all_values", "w") as f:
        f.write("Interpolated regressors and dependent varaibles:\n\n")
        f.write(combined_df.to_string(index=False))

    df = combined_df.rename(columns={'variable': 'Variable'})

    #Poverty gap index
    """df=FunctionsExtensions.load_data(file_path_Poverty_gap)
    year_cols = list(range(1989, 2024))  # or use integers if that's your format

    df=FunctionsExtensions.filter_top_countries_by_data_count_wide(df, country_col='Country', top_n=40, year_cols= year_cols)
    df_interpolated = FunctionsExtensions.interpolate_wide_pchip(df, country_col='Country', variable_col='Variable',
                           start_year=1989, end_year=2023)
    with open("Poverty_gap_interpolated.txt", "w") as f:
        f.write("Interpolated poverty gap index values per country:\n\n")
        f.write(df_interpolated.to_string(index=False))"""

    year_columns = list(range(1989, 2023))
    #df=FunctionsExtensions.standardize_panel_wide(df, country_col='Country', variable_col='Variable')

    # Step 1: Load and reshape the panel data
    df = FunctionsLatentTimeVarying.load_panel_data_pov_gap(df, year_columns)
    print(df)

    '''regressors_list = ['Government_expenditure', 'Human_Development_Index', 'Political_corruption']
    y_var=['Poverty_gap']

    regressors_list = ['Government_expenditure', 'Political_corruption']
    y_var = 'Human_Development_Index

    regressors_list = ['Government_expenditure', 'Political_corruption','Human_Development_Index']
    y_var = 'Gini_coefficient'''


    # Example: assuming df is your DataFrame
    # df = pd.read_csv('your_data.csv')  # or however you're loading your data

    regressors_list = [ 'Government_expenditure',
                       'Human_Development_Index', 'Political_corruption']
    y_var = 'Poverty_gap'

    # Create new interaction terms with Gini
    for var in regressors_list:
        if var != 'Gini_coefficient':
            new_col = f'Gini_times_{var}'
            df[new_col] = df['Gini_coefficient'] * df[var]

    # Optional: if you want to keep only the relevant columns or check them
    print(df.head())

    regressors_list = ['Government_expenditure', 'Human_Development_Index']
    y_var = 'Poverty_gap'
    regressors_list = ['Gini_coefficient','Government_expenditure', 'Human_Development_Index', 'Political_corruption']
    y_var = 'Poverty_gap'

    regressors_list = [ 'Gini_times_Government_expenditure', 'Gini_times_Human_Development_Index', 'Gini_times_Political_corruption']
    y_var = 'Poverty_gap'

    intercept='Country mean (intercept)'
    x_limit=(1986,2024)
    d=len(regressors_list)+1

    Y_panel, X_list, SX, SY, YV, vt, VT, u, h, N, T, pivoted, d = FunctionsExtensions.prepare_panel_data_with_dynamic_names_unpacked(
        df,
        y_var,
        regressors_list,
        start_year=1989,
        end_year=2022,
        h=0.1, d=d
    )
    unique_countries = df['Country'].drop_duplicates()


    print(SX.shape)
    print(YV.shape)
    print(X_list[0].shape)

    # Plotting
    var_names = [y_var] + regressors_list
    FunctionsExtensions.plot_panel_variables(YV, Y_panel, X_list, var_names, xlim=x_limit, ncols=d)

    # Step 5: Apply kernel smoothing
    coef_estimates = FunctionsLatentTimeVarying.cv_ksmooth(X_list, VT, Y_panel, u, h)

    # Step 6: Label and display results
    row_labels = [intercept] + regressors_list
    country_names = pivoted[y_var].columns.tolist()
    coef_df = pd.DataFrame(coef_estimates, index=row_labels, columns=country_names)

    print("Estimated coefficients (rows = predictors, columns = countries):")

    with open("CVKsmooth_specific.txt", "w") as f:
        f.write("Estimated coefficients (rows = predictors, columns = countries):\n")
        f.write(coef_df.to_string())  # Convert DataFrame to string for writing

    SX = np.asarray(SX, dtype=np.float64)
    SY = np.asarray(SY, dtype=np.float64)
    vt = np.asarray(vt, dtype=np.float64)

    coef = FunctionsLatentTimeVarying.find_coef(X_list, Y_panel, VT, h, d)
    with open("coefficients_overall.txt", "w") as f:
        f.write("Coefficients evaluated at each u :\n")
        f.write(np.array2string(coef, threshold=np.inf))

    result = FunctionsLatentTimeVarying.remove_fixed_effects(X_list, Y_panel, VT, SX, SY, vt, h)
    result_T_by_N = result.reshape(T, N)  # Replace T and N with actual values
    with open("removed_fixed_effects.txt", "w") as f:
        f.write("Fixed effect-removed result:\n")
        f.write(np.array2string(result, threshold=np.inf))

    # select the best bandwidth
    k_fold = T
    MY, CVSE, hopt = FunctionsLatentTimeVarying.hselector(X_list, Y_panel, VT, SX, SY, vt, T, h_len=10)

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
    SY_removed_effects = FunctionsLatentTimeVarying.remove_fixed_effects(X_list, Y_panel, VT, SX, SY, vt, hopt)
    with open("removed_effects_depend.txt", "w") as f:
        f.write(" evaluated at each u in N by 3T dimension:\n")
        f.write(np.array2string(SY_removed_effects, threshold=np.inf))

    Y_panel_removed_effects = SY_removed_effects.reshape((T, N), order='F')  # 'F' for column-major order
    with open("panel_removed_effects_coefficients.txt", "w") as f:
        f.write("Coefficients evaluated at each u in N by 3T dimension:\n")
        f.write(np.array2string(Y_panel_removed_effects, threshold=np.inf))

    coef = FunctionsLatentTimeVarying.find_coef(X_list, Y_panel_removed_effects, VT, hopt, d)

    # Now reshape to (N, 3T) by stacking the T rows horizontally
    Ebeta = coef.reshape(T * d, N, order='F').T
    with open("reshaped_coefficients.txt", "w") as f:
        f.write("Coefficients evaluated at each u in N by 3T dimension:\n")
        f.write(np.array2string(Ebeta, threshold=np.inf))

    # Assuming SY, SX, Ebeta, M, and T are already defined numpy arrays or variables
    # Reshape Ebeta similar to the MATLAB indexing
    M = N * T

    # Stack the reshaped parts horizontally (i.e., columns)
    reshaped_Ebeta = coef
    with open("reshaped_Ebeta.txt", "w") as f:
        f.write("Coeffficients reshaped:\n")
        f.write(np.array2string(reshaped_Ebeta, threshold=np.inf))

    FunctionsLatentTimeVarying.compute_np_mse(SX, reshaped_Ebeta, SY_removed_effects, parts_filename="parts",
                                              row_sums_filename="row_sums")

    # Compute the distance between pairwise points
    D = FunctionsLatentTimeVarying.dist_func(SX, SY, reshaped_Ebeta, hopt, vt)
    print(D)

    with open("distances_ebeta", "w") as f:
        f.write("Pairwise distances between coefficients:\n")
        for row in D:
            f.write(" ".join(str(value) for value in row) + "\n")

    """with open("distances_ebeta", "w") as f:
        f.write("Condensed pairwise distances between coefficients:\n")
        f.write(" ".join(str(value) for value in distances) + "\n")"""""

    K_clusters = 4
    Z, mship = FunctionsLatentTimeVarying.kcluster(N, K_clusters, D)

    result = FunctionsLatentTimeVarying.membership(SX, SY_removed_effects, vt, mship, K_clusters, hopt)
    Ealpha = result['Ealpha']
    mship = result['mship']
    EG = result['EG']
    EG_len = result['EG_len']
    Ealpha_inv = result['Ealpha_inv']

    with open('membership_result.txt', 'w') as f:
        for key, value in result.items():
            f.write(f"{key}:\n{value}\n\n")

    ebeta_groups = FunctionsLatentTimeVarying.find_ebeta_groups(Ealpha_inv, mship, N, T, d)
    with open("Ebeta_groups.txt", "w") as f:
        f.write("Coeffficients groups ebeta reshaped:\n")
        f.write(np.array2string(ebeta_groups, threshold=np.inf))
    FunctionsLatentTimeVarying.compute_np_mse(SX, ebeta_groups, SY_removed_effects, parts_filename="parts_groups",
                                              row_sums_filename="row_sums_groups")

    min_NK, AICrho_tuning, BICrho_tuning, variation = FunctionsLatentTimeVarying.variation(X_list,
                                                                                           Y_panel_removed_effects, VT,
                                                                                           SX, SY_removed_effects, vt,
                                                                                           hopt, 4)

    with open("variation.txt", "w") as f:
        f.write(f"return {min_NK}, {AICrho_tuning}, {BICrho_tuning}, {variation}\n")

    AICK, BICK, [BICKhat, AICKhat] = FunctionsLatentTimeVarying.Kselector(X_list, Y_panel_removed_effects, VT, SX,
                                                                          SY_removed_effects, vt, hopt)
    with open("Kselector.txt", "w") as f:
        f.write(f"return {AICK}, {BICK}, {[BICKhat, AICKhat]}\n")
    Z_AIC, mship_AIC = FunctionsLatentTimeVarying.kcluster(N, AICKhat, D)
    Z_BIC, mship_BIC = FunctionsLatentTimeVarying.kcluster(N, BICKhat, D)

    result_ = FunctionsLatentTimeVarying.membership(SX, SY_removed_effects, vt, mship_AIC, AICKhat, hopt)
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

    with open('BICmship.txt', 'w') as f:
        for item in BICmship:
            f.write(str(item) + '\n')

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
    with open("groups_countries_table_poverty_gap.tex", "w") as f:
        f.write(latex_table)

    ebeta_groups = FunctionsLatentTimeVarying.find_ebeta_groups(BICEalpha_inv, BICmship, N, T, d)
    print(d)
    FunctionsLatentTimeVarying.plot_group_coefficients(YV, BICEalpha, T, BICKhat, num_regresors=d, xlim=x_limit,
                                                       ylim_list=None)
    YX_list = [Y_panel] + X_list[1:]
    FunctionsLatentTimeVarying.plot_grouped_time_series(YV, YX_list, BICEG, BICKhat, var_names, xlim=x_limit,
                                                        num_regressors_dependent=d)


if __name__ == '__main__':
   main()