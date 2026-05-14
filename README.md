# Latent Groups with Time-Varying Coefficients
### July 7, 2025
## Overview

This repository contains the implementation and empirical/simulation analysis for a model of **latent group structures with time-varying coefficients** applied to panel data.  

The project investigates how heterogeneous group structures and evolving coefficients can better capture dynamics in economic and social indicators compared to standard fixed-effects or homogeneous panel models.

The framework is applied to multiple datasets, including economic growth, poverty measures, and development indicators.

---

## Key Features

- Estimation of latent group membership in panel data
- Time-varying coefficient modeling
- Group-specific parameter dynamics
- Simulation-based evaluation of model performance
- Bandwidth selection and smoothing procedures
- Model comparison using BIC and cross-validation criteria
- Application to real-world socio-economic datasets


# Latent Groups with Time-Varying Coefficients

## Overview

This repository contains the implementation and empirical/simulation analysis for a model of **latent group structures with time-varying coefficients** applied to panel data.  

The project investigates how heterogeneous group structures and evolving coefficients can better capture dynamics in economic and social indicators compared to standard fixed-effects or homogeneous panel models.

The framework is applied to multiple datasets, including economic growth, poverty measures, and development indicators.

---

## Key Features

- Estimation of latent group membership in panel data
- Time-varying coefficient modeling
- Group-specific parameter dynamics
- Simulation-based evaluation of model performance
- Bandwidth selection and smoothing procedures
- Model comparison using BIC and cross-validation criteria
- Application to real-world socio-economic datasets

---

## Project Structure

```text
.
├── mainEconomicGrowth.py
├── mainPovertyGap.py
├── mainDGP1_logistic.py
├── mainDGP2_Tuckey.py
│
├── FunctionsLatentTimeVarying.py
├── FunctionsExtensions.py
├── Fun.py
│
├── Data/
│   ├── Gini_coefficient.xlsx
│   ├── Government_expenditure.xlsx
│   ├── Human_Development_Index.xlsx
│   ├── Poverty_gap.xlsx
│   └── WDI_EconomicGrowth.xlsx
│
├── results/
│   ├── simulation_RMSE.txt
│   ├── summary_results_Logistic.txt
│   ├── summary_results_Tuckey.txt
│   └── membership_result*.txt
│
├── requirements.txt
└── README.md
````

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gabriela2002new/Bachelor-s-Thesis-Gabriela-Cretu.git
cd Bachelor-s-Thesis-Gabriela-Cretu
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Economic Growth analysis

```bash
python mainEconomicGrowth.py
```

### Poverty Gap analysis

```bash
python mainPovertyGap.py
```

### Simulation: Logistic DGP

```bash
python mainDGP1_logistic.py
```

### Simulation: Tukey-based DGP

```bash
python mainDGP2_Tuckey.py
```

---

## Methodology Summary

The model combines:

* **Latent group identification** in panel data
* **Time-varying coefficient estimation**
* **Nonparametric smoothing techniques**
* **Model selection via information criteria (BIC)**
* **Cross-validation for tuning parameters**

The approach allows capturing:

* structural heterogeneity across countries
* temporal evolution of effects
* nonlinear dynamics in economic indicators

---

## Data Sources

* World Development Indicators (WDI)
* Human Development Index (HDI)
* Government expenditure datasets
* Poverty and inequality measures (Gini, poverty gap)

---

## Output Examples

The repository generates:

* Group membership assignments
* Estimated coefficient trajectories
* Simulation error metrics (RMSE)
* Model comparison tables (BIC, CV scores)

---

## Requirements

Main dependencies:

* numpy
* pandas
* scipy
* scikit-learn
* matplotlib

(Full list in `requirements.txt`)

---

## Notes

* This project is research-oriented and designed for reproducibility.
* Some outputs are precomputed and stored in `/results`.
* Large datasets are included in `/Data` for full replication.

---

## Author

Gabriela Crețu

---

## License

For academic and research use only.
