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
    
    # Loading data
    TEST_DATA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/TestingModels/FakeData/fake_data_c=0.05.pkl"
    TEST_WEIGHTS_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/TestingModels/FakeData/fake_weights_c=0.05.pkl"
    MODEL_DIRECTORY = "/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/Results/FullScoreTest"
    
    # Load and format the test data
    with open(TEST_DATA_PATH, "rb") as datafile:
        data = pkl.load(datafile)
    kinematics = data.to_numpy()[:, 1:]
    # Load and format the test weights
    with open(TEST_WEIGHTS_PATH, "rb") as datafile:
        weights = pkl.load(datafile)
    comparison_coefficients = [string_to_float(column) if not "sm" in column else 0.0 for column in weights.columns]
    
    print(comparison_coefficients)
    
    # Load all the model paths from the directory into a dictionary
    files = glob.glob(MODEL_DIRECTORY + "/*/")
    titles = [list(filter(lambda item: item != "", file.split("/")))[-1] for file in files]
    
    # Iteratively evaluate each model on the test data and store its performance metrics and some plots
    performances = {}
    for file, title in zip(files, titles):
        # Get the paths to the alpha and beta models
        alpha_path = next(filter(lambda name: "alpha" in name, glob.glob(file + "/*/")), None)
        beta_path = next(filter(lambda name: "beta" in name, glob.glob(file + "/*/")), None)
        
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
        
        # Get the ratio estimates
        ratio_estimates = np.asarray([ratio_estimate(kinematics, coefficient) for coefficient in comparison_coefficients]).T
        # Get the actual ratios
        true_ratios = np.asarray(weights) / np.asarray(weights)[:, 0:1]
        
        # Chi squared calculation
        ratio_residuals = ratio_estimates - true_ratios
        
        chi_squared = np.sum(ratio_residuals ** 2.0, axis=0)
        print(chi_squared)
        
        
        ####### TODO
        # Store chi_squared, maybe residuals in dictionary
        # Compare them between models
        # Maybe test confidence intervals directly
    
    # Display the performance metrics for each model and make some plots to compare them
    
    
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