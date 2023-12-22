"""
# All Training Utils

---------------------------------------------------------------------------------------------------------------------------------

Kye Emond

August 31, 2023

---------------------------------------------------------------------------------------------------------------------------------

A script designed to use a set of events and their weights for a variety of Wilson Coefficient values to generate a realistic \
dataset. 
"""

import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt
import keras as k
import matplotlib.pyplot as plt
from all_training_utils import float_to_string, string_to_float
from analysis_utils import weighted_mean, weighted_var

def main() -> None:
    
    print("Start")
    
    DATA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/all_data.pkl"
    C_PREFIX = "weight_cHW_"
    C_VALUE = 0.05
    KINEMATIC_COLUMNS = np.arange(2, 39)
    ALPHA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Results/MSEDirect_8_layers/cHW_MSEDirect_8_layers_n_alpha"
    BETA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Results/MSEDirect_8_layers/cHW_MSEDirect_8_layers_n_beta"
    TESTING_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots"
    
    data = pd.read_pickle(DATA_PATH)
    
    for name in data.columns:
        if "weight_" in name:
            data[name] *= data["weight"]
    
    print(data.to_numpy().shape)
    
    weights = (data[C_PREFIX + float_to_string(C_VALUE)] if C_VALUE != 0.0 else data["weight_sm"]).to_numpy()
    
    # SEED = 122807528840384100672342137670123435476 + n
    # # rng = np.random.default_rng(seed=122807528840384100672342137670123456789)
    # rng = np.random.default_rng(seed=SEED)
    # n_samples = (rng.poisson(np.abs(weights)) * np.sign(weights)).astype(int)
    
    # indices = np.concatenate([[index] * number for index, number in enumerate(np.abs(n_samples).astype(int))]).astype(int)
    
    # fake_data = np.concatenate((np.sign(n_samples)[indices][:, np.newaxis], data.to_numpy()[indices][:, KINEMATIC_COLUMNS]), axis=1)
    
    ## TEMP STUFF
    
    n_alpha = k.models.load_model(ALPHA_PATH, compile=False)
    n_beta = k.models.load_model(BETA_PATH, compile=False)
    def best_dcross_section_ratio(c: npt.ArrayLike, data: npt.ArrayLike) -> np.ndarray:
        """Return the differential cross-section ratio (DCSR) estimates from the neural network.

        Args:
            * c (npt.ArrayLike): A 1D array_like that contains the values of the Wilson Coefficient for which to evaluate DCSRs.
            * data (npt.ArrayLike): A 2D array_like of kinematic variables. Individual events should be arranged along axis 0 and \
                different variables should be arranged along axis 1.

        Returns:
            * np.ndarray: A 2D array of the estimated DCSR for each element in `c` and each event in `data`. Events are arranged \
                along axis 0 and Wilson Coefficient values are arranged along axis 1.
        """
        
        print("Using Network for ratio estimation")
        return ((1.0 + np.ravel(c) * np.squeeze(n_alpha(data))[:, np.newaxis]) ** 2.0 
                + (np.ravel(c) * np.squeeze(n_beta(data))[:, np.newaxis]) ** 2.0)
    
    ## TEMP STUFF
    # SEED = 122807528840384100672342137670123475463
    
    dcsr_true = data["weight_sm"].to_numpy()/ weights
    dcsr_estimate = best_dcross_section_ratio(0.05, data.to_numpy()[:, KINEMATIC_COLUMNS])
    residuals = dcsr_estimate - dcsr_true
    plt.hist(residuals)
    plt.xlabel("Residuals")
    plt.ylabel("Count")
    plt.savefig("ResidualHist.png")
    exit()
    
    coefs = []
    for n in range(0, 22, 2):
        SEED = 122807528840384100672342137670123435476 + n
        # rng = np.random.default_rng(seed=122807528840384100672342137670123456789)
        rng = np.random.default_rng(seed=SEED)
        n_samples = (rng.poisson(np.abs(weights)) * np.sign(weights)).astype(int)
        
        indices = np.concatenate([[index] * number for index, number in enumerate(np.abs(n_samples).astype(int))]).astype(int)
        
        # fake_data = np.concatenate((np.sign(n_samples)[indices][:, np.newaxis], data.to_numpy()[indices][:, KINEMATIC_COLUMNS]), axis=1)
        
        ## MORE TEMP
        
        smaller_data = data.iloc[indices]
        
        kinematics = smaller_data.to_numpy()[:, KINEMATIC_COLUMNS]
        
        means = []
        mean_uncert = []
        cs = (-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1)
        for c in cs:
            test_weights = (smaller_data[C_PREFIX + float_to_string(c)] if c != 0.0 else smaller_data["weight_sm"]).to_numpy()
            dcsr = test_weights / smaller_data["weight_sm"].to_numpy()
            
            log_difference = np.squeeze(np.log(dcsr)) - np.squeeze(np.log(best_dcross_section_ratio(c, kinematics)))
            means.append(weighted_mean(log_difference, np.sign(test_weights)))
            mean_uncert.append(np.sqrt(weighted_var(log_difference, np.sign(test_weights)) / len(dcsr)))
            print(c, weighted_mean(np.squeeze(np.log(dcsr)), np.sign(test_weights)), weighted_mean(log_difference, np.sign(test_weights)), len(dcsr) * weighted_mean(log_difference, np.sign(test_weights)), np.sqrt(len(dcsr) * weighted_var(log_difference, np.sign(test_weights))))
            print(100.0 * weighted_mean(log_difference, np.sign(test_weights)) / weighted_mean(np.squeeze(np.log(dcsr)), np.sign(test_weights)))
            # plt.hist(log_difference, bins=100)
            # plt.title(f"{weighted_mean(log_difference, test_weights)}, {weighted_std(log_difference, test_weights)}")
            # plt.savefig(TESTING_PATH + f"/histogram_{c}.png")
        
        means = np.asarray(means)
        mean_uncert = np.asarray(mean_uncert)
        cs = np.asarray(cs)
        
        mean_coefs = npp.polyfit(cs, means, deg=2)
        print("Coefs", mean_coefs)
        coefs.append(mean_coefs)
        print(coefs)
        def mean_func(c: npt.ArrayLike, num: float) -> np.ndarray:
            return npp.polyval(c, mean_coefs) * num
        
        plt.errorbar(cs, means, mean_uncert, fmt=".")
        plt.plot(np.linspace(-0.1, 0.1, 100), mean_func(np.linspace(-0.1, 0.1, 100), 1), alpha=0.2)
    plt.savefig(TESTING_PATH + f"/means_{C_VALUE}_{SEED}.png")
    print("Coefficients")
    print(coefs)
    print("Average")
    print(np.mean(coefs, axis=0))
    print("done")
    
    ## MORE TEMP
    
    
    # print(fake_data.shape)
    # # print(fake_data)
    
    # print(np.sum(weights))
    # print(np.sum(n_samples))
    # print(np.sum(fake_data[:, 0]))
    
    # fake_data_pd = pd.DataFrame(fake_data)
    
    # print(fake_data_pd)
    
    # fake_data_pd.to_pickle("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/AnalysisTest/fake_inference0.pkl")
    
    

if __name__ == "__main__":
    main()