import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from scipy.interpolate import PchipInterpolator

import pandas as pd
import re
import FunctionsLatentTimeVarying

valid_countries = {
    'afghanistan', 'albania', 'algeria', 'andorra', 'angola', 'argentina', 'armenia',
    'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain', 'bangladesh', 'barbados',
    'belarus', 'belgium', 'belize', 'benin', 'bhutan', 'bolivia', 'bosnia and herzegovina',
    'botswana', 'brazil', 'brunei', 'bulgaria', 'burkina faso', 'burundi', 'cambodia',
    'cameroon', 'canada', 'cape verde', 'central african republic', 'chad', 'chile', 'china',
    'colombia', 'comoros', 'congo', 'costa rica', 'croatia', 'cuba', 'cyprus', 'czech republic',
    'denmark', 'djibouti', 'dominica', 'dominican republic', 'ecuador', 'egypt', 'el salvador',
    'equatorial guinea', 'eritrea', 'estonia', 'eswatini', 'ethiopia', 'fiji', 'finland',
    'france', 'gabon', 'gambia', 'georgia', 'germany', 'ghana', 'greece', 'grenada', 'guatemala',
    'guinea', 'guinea-bissau', 'guyana', 'haiti', 'honduras', 'hungary', 'iceland', 'india',
    'indonesia', 'iran', 'iraq', 'ireland', 'israel', 'italy', 'jamaica', 'japan', 'jordan',
    'kazakhstan', 'kenya', 'kiribati', 'kuwait', 'kyrgyzstan', 'laos', 'latvia', 'lebanon',
    'lesotho', 'liberia', 'libya', 'liechtenstein', 'lithuania', 'luxembourg', 'madagascar',
    'malawi', 'malaysia', 'maldives', 'mali', 'malta', 'marshall islands', 'mauritania',
    'mauritius', 'mexico', 'micronesia', 'moldova', 'monaco', 'mongolia', 'montenegro',
    'morocco', 'mozambique', 'myanmar', 'namibia', 'nauru', 'nepal', 'netherlands',
    'new zealand', 'nicaragua', 'niger', 'nigeria', 'north korea', 'north macedonia', 'norway',
    'oman', 'pakistan', 'palau', 'panama', 'papua new guinea', 'paraguay', 'peru', 'philippines',
    'poland', 'portugal', 'qatar', 'romania', 'russia', 'rwanda', 'saint kitts and nevis',
    'saint lucia', 'saint vincent and the grenadines', 'samoa', 'san marino',
    'sao tome and principe', 'saudi arabia', 'senegal', 'serbia', 'seychelles', 'sierra leone',
    'singapore', 'slovakia', 'slovenia', 'solomon islands', 'somalia', 'south africa',
    'south korea', 'south sudan', 'spain', 'sri lanka', 'sudan', 'suriname', 'sweden',
    'switzerland', 'syria', 'taiwan', 'tajikistan', 'tanzania', 'thailand', 'timor-leste',
    'togo', 'tonga', 'trinidad and tobago', 'tunisia', 'turkey', 'turkmenistan', 'tuvalu',
    'uganda', 'ukraine', 'united arab emirates', 'united kingdom', 'united states',
    'uruguay', 'uzbekistan', 'vanuatu', 'venezuela', 'vietnam', 'yemen', 'zambia', 'zimbabwe'
}

def filter_dfs_to_common_countries(dfs, country_col='Country'):
    # Get sets of countries directly
    country_sets = [set(df[country_col]) for df in dfs]
    common_countries = set.intersection(*country_sets)
    filtered_dfs = [df[df[country_col].isin(common_countries)].copy() for df in dfs]
    return filtered_dfs


def filter_countries(df, country_col='Country'):
    # Step 1: Remove rows with (urban) or (rural)
    urban_rural_mask = df[country_col].str.contains(r'\(.*(?:urban|rural).*?\)', case=False, regex=True)

    df = df[~urban_rural_mask].copy()

    # Step 2: Clean names and match to valid countries
    def clean_country_name(name):
        name = re.sub(r'\(.*?\)', '', name)  # Remove anything in parentheses
        return name.strip().lower()

    cleaned_names = df[country_col].apply(clean_country_name)
    valid_mask = cleaned_names.isin(valid_countries)
    df = df[valid_mask].copy()

    # Step 3: Replace country name with standardized version
    def standardize_name(name):
        name_lower = clean_country_name(name)
        for valid_country in valid_countries:
            if valid_country == name_lower:
                return valid_country.title()  # title-case it (e.g. "United States")
        return name  # fallback

    df[country_col] = df[country_col].apply(standardize_name)
    return df


# Example usage:
# Suppose your DataFrame is df with a 'Country' column
# df_cleaned = filter_countries(df, country_col='Country')

def load_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')

    return df


def interpolate_pchip(df, value_col='value', year_col='year', country_col='country',
                      return_start_year=1989, return_end_year=2023):
    # Use all available years in the data for interpolation
    min_year = min(df[year_col].min(), 1989)
    max_year = max(df[year_col].max(),2023)
    all_years = np.arange(min_year, max_year+1)

    # Years to include in the final output
    return_years = np.arange(return_start_year, return_end_year)

    records = []

    for country, group in df.groupby(country_col):
        group = group.sort_values(year_col)
        x = group[year_col].values
        y = group[value_col].values

        if len(x) < 2:
            y_interp_full = np.full(len(all_years), np.nan)
        else:
            # Enable extrapolation
            interpolator = PchipInterpolator(x, y, extrapolate=True)
            y_interp_full = interpolator(all_years)
            # Optional: clip negative values to 0
            y_interp_full = np.clip(y_interp_full, 0, None)

        # Build interpolation dictionary for all years
        interp_dict = dict(zip(all_years, y_interp_full))

        # Construct row with only return_years
        row = {
            country_col: country,
            'variable': value_col
        }
        row.update({year: float(interp_dict.get(year, np.nan)) for year in return_years})

        records.append(row)

    return pd.DataFrame(records)

def interpolate_wide_df_pchip(df, country_col='Country', variable_col='Variable',
                              start_year=1996, end_year=2023):
    """
    Interpolate missing values by country and variable in a DataFrame with years as columns.
    Automatically adds missing year columns in the range [start_year, end_year].

    Parameters:
    - df: DataFrame in wide format (columns include country_col, variable_col, and some years)
    - country_col: name of the country column
    - variable_col: name of the variable column
    - start_year: first year to include and interpolate
    - end_year: last year to include and interpolate

    Returns:
    - DataFrame with country_col, variable_col, and all years [start_year..end_year] as columns,
      with missing values interpolated using PCHIP.
    """

    # Define full list of years as columns (integers or strings)
    year_cols = [y for y in range(start_year, end_year + 1)]

    # Make sure all these year columns exist in df, add missing ones filled with NaN
    for year in year_cols:
        if year not in df.columns:
            df[year] = np.nan

    # We'll collect interpolated rows here
    interpolated_rows = []

    # Group by country and variable
    for (country, variable), group in df.groupby([country_col, variable_col]):
        # Extract the row (should be just one row per country-variable)
        row = group.iloc[0]

        # Extract values for all years as float, in order
        y = row[year_cols].astype(float).values

        # Identify known points for interpolation
        known_mask = ~np.isnan(y)
        known_years = np.array([int(y) for y in year_cols])[known_mask]
        known_values = y[known_mask]

        if len(known_values) < 2:
            # Not enough data to interpolate - keep original values
            interp_values = y
        else:
            # Interpolate for all years using PCHIP
            pchip = PchipInterpolator(known_years, known_values)
            interp_values = pchip(np.array([int(y) for y in year_cols]))

            # Clip negatives to zero if variable can't be negative (e.g., poverty gap)
            interp_values = np.clip(interp_values, 0, None)

        # Build the output row dict
        out_row = {
            country_col: country,
            variable_col: variable
        }
        out_row.update({year: val for year, val in zip(year_cols, interp_values)})

        interpolated_rows.append(out_row)

    return pd.DataFrame(interpolated_rows)
def filter_top_countries_by_data_count(df, country_col='country', year_col='year', top_n=200,min_y=1990):
    """
    Keep only the top_n countries with the most data points in the DataFrame,
    but only if the minimum year for each country is at least 1993.

    Parameters:
    - df: input DataFrame containing country_col and year_col
    - country_col: name of the column with country names
    - year_col: name of the column with years
    - top_n: number of countries to keep (default 60)

    Returns:
    - Filtered DataFrame containing only data for the top_n countries by data count
      where the minimum year is at least 1993
    """
    # Count number of records per country
    country_counts = df[country_col].value_counts()

    # Get list of countries ordered by count
    countries_by_count = country_counts.index.tolist()

    # Filter countries to keep only those with min year >= 1993
    valid_countries = []
    for country in countries_by_count:
        min_year = df.loc[df[country_col] == country, year_col].min()
        max_year = df.loc[df[country_col] == country, year_col].max()
        if min_year <= min_y and country_counts[country]>25:
            valid_countries.append(country)
        if len(valid_countries) == top_n:
            break

    # Filter original df to keep only those countries
    filtered_df = df[df[country_col].isin(valid_countries)].copy()

    return filtered_df


def filter_top_countries_by_data_count_wide(df, country_col='Country', top_n=60, year_cols=None):
    """
    Keep only the top_n countries with the most non-NaN data points across all variables and years.

    Parameters:
    - df: input DataFrame in wide format (e.g., with columns: country, variable, 1996–2023)
    - country_col: name of the column with country names
    - top_n: number of countries to keep (default 60)
    - year_cols: list of columns that correspond to years

    Returns:
    - Filtered DataFrame containing only data for the top_n countries by data coverage
    """
    if year_cols is None:
        raise ValueError("Please provide the year_cols (list of year column names).")

    # Count non-NaN entries per country across all year columns and all variables
    data_counts = df.groupby(country_col)[year_cols].apply(lambda x: x.notna().sum().sum())

    # Select top_n countries with most non-NaN values
    top_countries = data_counts.nlargest(top_n).index.tolist()

    # Filter DataFrame to include only those countries
    filtered_df = df[df[country_col].isin(top_countries)].copy()

    return filtered_df

def standardize_panel_wide(df, country_col='country', variable_col='Variable'):
    # Identify year columns (assume all columns except country and Variable are years)
    year_cols = [col for col in df.columns if col not in [country_col, variable_col]]

    # Function to standardize a row (values across years)
    def standardize_row(row):
        vals = row[year_cols].values.astype(float)
        mean = vals.mean()
        std = vals.std(ddof=0)
        if std == 0:
            return pd.Series([0]*len(vals), index=year_cols)  # or np.nan if preferred
        else:
            return pd.Series((vals - mean) / std, index=year_cols)

    # Apply per row
    df_std = df.copy()
    df_std[year_cols] = df_std.apply(standardize_row, axis=1)

    return df_std


def prepare_panel_data_with_dynamic_names_unpacked(df, y_var, x_vars, country_col='Country', year_col='Year',
                                                  start_year=None, end_year=None, h=0.1,d=4):

    variables = [country_col, year_col, y_var] + x_vars
    df_model = df[variables].dropna()
    pivoted = df_model.pivot(index=year_col, columns=country_col)
    Y_panel = pivoted[y_var].values
    T, N = Y_panel.shape

    X1 = np.ones_like(Y_panel)
    regressors = [pivoted[var].values for var in x_vars]

    mask = ~np.isnan(X1) & ~np.isnan(Y_panel)
    for reg in regressors:
        mask &= ~np.isnan(reg)
    X1[~mask] = 0
    for reg in regressors:
        reg[~mask] = 0
    Y_panel[~mask] = 0

    X_list = [X1] + regressors

    reg_flattened = [reg.flatten(order='F') for reg in regressors]
    ones = np.ones((T * N, 1))
    if reg_flattened:
        SX = np.hstack((ones, np.column_stack(reg_flattened)))
    else:
        SX = ones

    SY = Y_panel.flatten(order='F').reshape(-1, 1)

    if start_year is not None and end_year is not None:
        YV = np.arange(start_year, end_year + 1)
    else:
        YV = pivoted.index.values

    vt = np.arange(1, T + 1).reshape(T, 1) / T
    VT = np.tile(vt, N)
    u = np.tile((1 / T), N)
    N = N
    T = T

    # Return variables unpacked:
    # Y_panel, X1, X2, ..., SX, SY, YV, vt, VT, u, h, N, T, pivoted_df
    return Y_panel, X_list, SX, SY, YV, vt, VT, u, h, N, T, pivoted,d

import matplotlib.pyplot as plt
import math

def plot_panel_variables(YV, Y_panel, X_list, var_names, xlim=(1988, 2023), ncols=4):
    """
    Plot dependent variable and regressors from panel data.

    Parameters:
    - YV: array-like of years (length T)
    - Y_panel: 2D array (T x N) for dependent variable
    - X_list: list of 2D arrays (T x N), first is intercept (skipped in plot)
    - var_names: list of strings, names of variables starting with dependent variable,
                 then names of regressors excluding intercept
    - xlim: tuple for x-axis limits
    - ncols: number of subplot columns

    Returns:
    - None (shows plot)
    """
    nplots = 1 + (len(X_list) - 1)  # dep var + regressors (skip intercept)
    nrows = math.ceil(nplots / ncols)

    plt.figure(figsize=(5 * ncols, 4 * nrows))

    # Plot dependent variable in first subplot
    FunctionsLatentTimeVarying.plot_panel_subplot(YV, Y_panel, (nrows, ncols, 1), var_names[0], xlim=xlim)

    # Plot regressors from X_list[1:] (skip intercept)
    for i, (X, name) in enumerate(zip(X_list[1:], var_names[1:]), start=2):
        FunctionsLatentTimeVarying.plot_panel_subplot(YV, X, (nrows, ncols, i), name, xlim=xlim)

    plt.tight_layout()
    plt.show()


import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_panel_by_country_variable(df, country_col='Country', variable_col='Variable'):
    """
    Scales year-wise panel data per (country, variable) pair using z-score normalization.

    Parameters:
        df (pd.DataFrame): The input DataFrame in wide format where each row is a (Country, Variable)
                           pair and columns are years.
        country_col (str): Name of the country column.
        variable_col (str): Name of the variable column.

    Returns:
        pd.DataFrame: Scaled DataFrame with the same structure.
    """
    df_scaled = df.copy()
    # Identify year columns (exclude country and variable columns)
    year_columns = [col for col in df.columns if col not in [country_col, variable_col]]

    # Apply scaling per (country, variable) pair
    for (country, var), group in df.groupby([country_col, variable_col]):
        values = group[year_columns].values.astype(float)
        scaled_values = StandardScaler().fit_transform(values)
        df_scaled.loc[group.index, year_columns] = scaled_values

    return df_scaled







