"""
# Comparing Models

---------------------------------------------------------------------------------------------------------------------------------

Kye Emond

Janurary 10, 2024

---------------------------------------------------------------------------------------------------------------------------------

A script you can run to compare the accuracy of models under a given directory
"""

# Imports
import numpy as np
import glob as glob
import pickle as pkl
# import keras as k
# import tensorflow as tf
import keras.models as km
import matplotlib.pyplot as plt
import typing as ty
import numpy.typing as npt
import matplotlib.pyplot as plt

# import all_training_utils as at

# Main function
def main() -> None:
    
    print("\n" + "START".center(50, "-"))
    
    # Loading data
    TEST_DATA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/TestingModels/FakeData/fake_data_c=-0.05.pkl"
    TEST_WEIGHTS_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/TestingModels/FakeData/fake_weights_c=-0.05.pkl"
    MODEL_DIRECTORY = "/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/Results/FullScoreTest"
    
    print("\n" + "LOADING DATA".center(50, "-"))
    
    # Load and format the test data
    with open(TEST_DATA_PATH, "rb") as datafile:
        data = pkl.load(datafile)
    kinematics = data.to_numpy()[:, 1:]
    # Load and format the test weights
    with open(TEST_WEIGHTS_PATH, "rb") as datafile:
        weights = pkl.load(datafile)
    comparison_coefficients = [string_to_float(column) if not "sm" in column else 0.0 for column in weights.columns]
    
    print("\n" + "SEARCHING FOR MODELS".center(50, "-"))
    
    # Load all the model paths from the directory into a dictionary
    files = glob.glob(MODEL_DIRECTORY + "/*/")
    titles = [list(filter(lambda item: item != "", file.split("/")))[-1] for file in files]
    
    print("\n" + "EVALUATING PERFORMANCES".center(50, "-"))
    
    # Iteratively evaluate each model on the test data and store its performance metrics and some plots
    performances = {}
    for file, title in zip(files, titles):
        try:
            print("\n" + f"STARTING {title}".center(50, "-"))
            
            # Get the paths to the alpha and beta models
            alpha_path = next(filter(lambda name: "alpha" in name, glob.glob(file + "/*/")), None)
            beta_path = next(filter(lambda name: "beta" in name, glob.glob(file + "/*/")), None)
            
            print("\n" + "LOADING MODELS".center(50, "-"))
            
            # Load the submodels
            alpha = km.load_model(alpha_path, compile=False)
            beta = km.load_model(beta_path, compile=False)
            
            # Design the ratio estimate function
            def ratio_estimate(kinematics: npt.ArrayLike, coefficient: float) -> np.ndarray:
                """Calculate the estimate for the differential cross-section ratio of a set of events at a given Wilson Coefficient. 

                Args:
                    kinematics (npt.ArrayLike): A 2D array storing kinematic variables along axis 1 and individual events along\
                        axis 0. 
                    coefficient (float): The Wilson Coefficient value at which to evaluate the ratio estimate. 

                Returns:
                    np.ndarray: A 1D array of the ratio estimates for each event. 
                """
                return (1.0 + coefficient * alpha(kinematics)[..., 0]) ** 2.0 + (coefficient * beta(kinematics)[..., 0]) ** 2.0
            
            print("\n" + "CALCULATING ESTIMATES".center(50, "-"))
            
            # Get the ratio estimates
            ratio_estimates = np.asarray([ratio_estimate(kinematics, coefficient) for coefficient in comparison_coefficients]).T
            # Get the actual ratios
            true_ratios = np.asarray(weights) / np.asarray(weights)[:, 0:1]
            
            print("\n" + "CALCULATING CHI-SQUARED".center(50, "-"))
            
            # Chi squared calculation
            ratio_residuals = ratio_estimates - true_ratios
            chi_squared = np.sum(ratio_residuals ** 2.0, axis=0)
            
            performances[title] = chi_squared
            # performances[title] = np.sum(ratio_residuals, axis=0)
            # print(performances)
            
            ####### TODO
            # Store chi_squared, maybe residuals in dictionary
            # Compare them between models
            # Maybe test confidence intervals directly
        except:
            print("ERROR - probably no filepath for this guy")
    
    print("\n" + "PLOTTING".center(50, "-"))
    # Display the performance metrics for each model and make some plots to compare them
    max_name = ""
    min_name = ""
    max_vals = np.zeros(len(comparison_coefficients))
    min_vals = performances["MSEFRACpos1p0_SCORESUBFRACpos0p0"]
    plt.plot(comparison_coefficients, performances["MSEFRACpos1p0_SCORESUBFRACpos0p0"], ".", label="MSEFRACpos1p0_SCORESUBFRACpos0p0")
    for name, values in performances.items():
        if (values - max_vals).mean() > 0.0:
            max_vals = values
            max_name = name
        elif (values - min_vals).mean() < 0.0:
            min_vals = values
            min_name = name
    plt.plot(comparison_coefficients, max_vals, ".", label=max_name)
    plt.plot(comparison_coefficients, min_vals, ".", label=min_name)
    plt.yscale("symlog")
    plt.legend()
    plt.xlabel("Tested Coefficient Value")
    plt.ylabel("Chi-Squared of Model")
    plt.title("Comparison of Different Models")
    plt.savefig("model_comparison_neg0p05.png")
    
    
def string_to_float(string: str) -> float:
    """Return the float corresponding to the weird nTuple-formatted string.

    Args:
        string (str): The formatted string to convert.

    Returns:
        float: The float version of the string
    """
    
    number_bit = string.split("_")[-1].replace("pos", "").replace("neg", "-").replace("p", ".")
    return float(number_bit)

# Call the main function
if __name__ == "__main__":
    main()