import pandas as pd
import numpy as np
import random
import FunctionsLatentTimeVarying  # your helper module
np.random.seed(1)  # Set seed once globally


def main():


    # Parameters
    K0 = 3
    N = 50
    T = 35
    R = 15
    vt = np.arange(1, T + 1) / T  # time values
    d = 2

    # Use your group membership function
    mship0 = FunctionsLatentTimeVarying.mship_logistic(N)  # shape (N,)

    # Initialize beta0 (N x 2T)

    # Fill beta0 using your generate_betas_logistic function
    Ealpha_0, Ebeta_0=FunctionsLatentTimeVarying.compute_beta0(mship0, vt,FunctionsLatentTimeVarying.generate_betas_logistic_group,normalize_first_beta=True)
    with open("beta_DGP1.txt", "w") as f:
        f.write("=== Group-Level Coefficients (Ealpha_0: shape {}): ===\n".format(Ealpha_0.shape))
        f.write(np.array2string(Ealpha_0, threshold=np.inf, precision=4))
        f.write("\n\n")

        f.write("=== Entity-Level Coefficients (Ebeta_0: shape {}): ===\n".format(Ebeta_0.shape))
        f.write(np.array2string(Ebeta_0, threshold=np.inf, precision=4))

    hopt=FunctionsLatentTimeVarying.monte_carlo_bandwidth(Ebeta_0, N, T, vt)
    print(hopt)
    results=FunctionsLatentTimeVarying.run_monte_carlo_np_estimation(mship0,Ealpha_0,Ebeta_0,N,T,vt,hopt,3,R=R,k_min=1,k_max=5)
    with open('simulation_RMSE.txt', 'w') as f:
        for key, value in results.items():
            f.write(f"{key}:\n")
            if isinstance(value, np.ndarray):
                f.write(np.array2string(value, threshold=np.inf, max_line_width=200))
            else:
                f.write(str(value))
            f.write("\n\n")

    # Extract results for summary stats
    NMI_rep_BIC = results['BIC']['NMI']
    NMI_rep_AIC = results['AIC']['NMI']
    Purity_rep_BIC = results['BIC']['Purity']
    Purity_rep_AIC = results['AIC']['Purity']
    Oracle_RMSE_rep_BIC = results['BIC']['Oracle_RMSE']
    RMSE_NP_rep_BIC = results['BIC']['RMSE_NP']
    RMSE_rep_BIC = results['BIC']['RMSE']
    RMSE_rep_AIC = results['AIC']['RMSE']

    # Compute summary statistics
    stats = FunctionsLatentTimeVarying.compute_summary_stats(
        NMI_rep_BIC, NMI_rep_AIC, Purity_rep_BIC, Purity_rep_AIC,
        Oracle_RMSE_rep_BIC, RMSE_NP_rep_BIC,
        RMSE_rep_BIC, RMSE_rep_AIC
    )

    # Extract frequency arrays
    Freq_BICK = results['BIC']['Freq']
    Freq_AICK = results['AIC']['Freq']

    Kmin = 1
    Kmax = 5
    with open('summary_results_Logistic.txt', 'w') as f:
        f.write(f"\nOptimal bandwidth used: h = {hopt}\n\n")
        f.write(f"BIC frequencies of cluster numbers {Kmin} - {Kmax}:\n")
        f.write(", ".join(map(str, Freq_BICK)) + "\n\n")

        f.write(f"AIC frequencies of cluster numbers {Kmin} - {Kmax}:\n")
        f.write(", ".join(map(str, Freq_AICK)) + "\n\n")

        f.write(f"BIC average NMI and std:\n{stats['ave_BICNMI']:.4f} ({stats['std_BICNMI']:.4f})\n\n")
        f.write(f"BIC average purity and std:\n{stats['ave_BICPurty']:.4f} ({stats['std_BICPurty']:.4f})\n\n")

        f.write(f"AIC average NMI and std:\n{stats['ave_AICNMI']:.4f} ({stats['std_AICNMI']:.4f})\n\n")
        f.write(f"AIC average purity and std:\n{stats['ave_AICPurty']:.4f} ({stats['std_AICPurty']:.4f})\n\n")

        f.write("Oracle RMSE (known clusters) average and std:\n")
        f.write(f"{stats['ave_RMSE_Orl']:.4f} ({stats['std_RMSE_Orl']:.4f})\n\n")

        f.write("Nonparametric RMSE average and std:\n")
        f.write(f"{stats['ave_RMSE_NP']:.4f} ({stats['std_RMSE_NP']:.4f})\n\n")

        f.write("BIC post-HAC RMSE average and std:\n")
        f.write(f"{stats['ave_BICRMSE_HAC']:.4f} ({stats['std_BICRMSE_HAC']:.4f})\n\n")

        f.write("AIC post-HAC RMSE average and std:\n")
        f.write(f"{stats['ave_AICRMSE_HAC']:.4f} ({stats['std_AICRMSE_HAC']:.4f})\n\n")


if __name__ == '__main__':
    main()
