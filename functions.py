import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from linearmodels.panel import PanelOLS


def load_panel_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')

    # The years are integers in your case
    year_columns = list(range(1971, 2017))
    id_vars = ['Country', 'Variable']

    # Melt to long format: one row per country-variable-year
    df_long = df.melt(id_vars=id_vars, value_vars=year_columns,
                      var_name='Year', value_name='Value')

    # Pivot so variables become columns
    df_pivot = df_long.pivot_table(index=['Country', 'Year'],
                                   columns='Variable', values='Value').reset_index()

    # Clean up column names
    df_pivot.columns.name = None
    df_pivot.columns = df_pivot.columns.str.strip()

    return df_pivot


def prepare_regression_data(df):
    df_clean = df.dropna(subset=['GDP growth', 'Population growth', 'Capital formation'])

    X = df_clean[['Population growth', 'Capital formation']].values
    Y = df_clean['GDP growth'].values
    meta = df_clean[['Country', 'Year']]

    # Create a combined DataFrame for viewing
    combined_df = meta.copy()
    combined_df['Population growth'] = df_clean['Population growth'].values
    combined_df['Capital formation'] = df_clean['Capital formation'].values
    combined_df['GDP growth'] = df_clean['GDP growth'].values

    print(combined_df.head())  # print first few rows

    return X, Y, meta



from linearmodels.panel import PanelOLS
import statsmodels.api as sm

def run_panel_regression(df_model):
    # Set panel index
    df_model = df_model.set_index(['Country', 'Year'])

    # Define independent and dependent variables
    exog = sm.add_constant(df_model[['Population growth', 'Capital formation']])
    endog = df_model['GDP growth']

    # Run fixed effects model
    model = PanelOLS(endog, exog, entity_effects=True)
    results = model.fit()

    print(results.summary)
    return results



def run_panel_ols(df):
    df_clean = df.dropna(subset=['GDP growth', 'Population growth', 'Capital formation'])

    # Set multi-index for panel data: entity=Country, time=Year
    df_clean = df_clean.set_index(['Country', 'Year'])

    exog_vars = ['Population growth', 'Capital formation']
    exog = sm.add_constant(df_clean[exog_vars])
    endog = df_clean['GDP growth']

    # Fit fixed effects (entity) model
    model = PanelOLS(endog, exog, entity_effects=True)
    results = model.fit()
    print(results.summary)

    return results

def run_countrywise_regressions(df_model):
    import statsmodels.api as sm

    results = {}
    for country, group in df_model.groupby('Country'):
        group = group.dropna()
        if len(group) < 5:  # Skip small samples
            continue
        X = sm.add_constant(group[['Population growth', 'Capital formation']])
        y = group['GDP growth']
        model = sm.OLS(y, X).fit()
        results[country] = model
        print(f"\nCountry: {country}")
        print(model.summary())

    return results

from statsmodels.regression.mixed_linear_model import MixedLM

def run_hierarchical_model(df_model):
    df_model = df_model.dropna(subset=['GDP growth', 'Population growth', 'Capital formation'])

    # Fit a random intercept + random slope model
    model = MixedLM.from_formula(
        'Q("GDP growth") ~ Q("Population growth") + Q("Capital formation")',
        groups='Country',
        re_formula='~Q("Population growth") + Q("Capital formation")',
        data=df_model
    )

    result = model.fit()
    print(result.summary())

    return result


import pymc as pm
import arviz as az
import pandas as pd
import numpy as np

def run_bayesian_hierarchical_model(df_model):
    df_model = df_model.dropna(subset=['GDP growth', 'Population growth', 'Capital formation']).copy()

    # Encode countries as integer groups
    df_model['Country_idx'] = df_model['Country'].astype('category').cat.codes
    country_idx = df_model['Country_idx'].values

    # Extract data
    y = df_model['GDP growth'].values
    X1 = df_model['Population growth'].values
    X2 = df_model['Capital formation'].values
    n_countries = df_model['Country_idx'].nunique()

    with pm.Model() as model:
        # Hyperpriors
        mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=10)
        mu_beta1 = pm.Normal('mu_beta1', mu=0, sigma=10)
        mu_beta2 = pm.Normal('mu_beta2', mu=0, sigma=10)

        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
        sigma_beta1 = pm.HalfNormal('sigma_beta1', sigma=1)
        sigma_beta2 = pm.HalfNormal('sigma_beta2', sigma=1)

        # Country-level effects
        alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_countries)
        beta1 = pm.Normal('beta1', mu=mu_beta1, sigma=sigma_beta1, shape=n_countries)
        beta2 = pm.Normal('beta2', mu=mu_beta2, sigma=sigma_beta2, shape=n_countries)

        # Expected value
        mu = alpha[country_idx] + beta1[country_idx] * X1 + beta2[country_idx] * X2

        # Likelihood
        sigma = pm.HalfNormal('sigma', sigma=1)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

        # Sampling
        trace = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True)

    # Summary
    az.plot_trace(trace, var_names=['mu_alpha', 'mu_beta1', 'mu_beta2', 'sigma'])
    az.summary(trace, var_names=['mu_alpha', 'mu_beta1', 'mu_beta2', 'sigma'])

    return trace
