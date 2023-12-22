"""
# Model Testing                                                                                                                 #
# Author: Kye Emond                                                                                                             #
# Date: June 15, 2023                                                                                                           #
#                                                                                                                               #
# A script to test the trained cross-section ratio models                                                                       #
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import typing as ty
import tensorflow as tf

from keras.models import load_model, Model
from glob import glob

from all_training_utils import events_to_training, float_to_string, build_loss


# Load the data
def main() -> None:

    
    # Constants
    DATA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/all_data.pkl"
    INTEREST_SLICE = slice(None, None)
    TRAINED_COEF_VALUES = (-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    C_VALUE_PREFIX = "weight_cHW_"
    INT_PREFIX = "weight_int_cHW_"
    BSM_PREFIX = "weight_bsm_cHW_"
    KINEMATIC_COLUMNS = np.arange(2, 39)#np.array((4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 28, 29, 30, 33, 35, 36, 37, 38))
    PATH_TO_MODEL = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Results/MSEDirect_8_layers"
    MODEL_NAME = "cHW_MSEDirect_8_layers"
    REMOVE_NEGATIVES = False

    COMPARISON_COEFS = (-0.5, 0.1, 0.5)
    SAVE_DIRECTORY = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Results/MSEDirect_8_layers/Plots"

    USE_WEIGHTS = False
    WEIGHT_COLUMNS = np.arange(39, 82)
    
    
    # Get the data
    events, kinematics, weights = get_data(filepath=DATA_PATH, 
                                           interest_slice=INTEREST_SLICE, 
                                           coef_values=TRAINED_COEF_VALUES, 
                                           randomize_data=False, 
                                           c_value_prefix=C_VALUE_PREFIX, 
                                           kinematic_columns=KINEMATIC_COLUMNS if not USE_WEIGHTS else WEIGHT_COLUMNS, 
                                           remove_negatives=REMOVE_NEGATIVES, 
                                           use_weights=USE_WEIGHTS)
    
    # Get the model
    quadratic, ratio_estimator, n_alpha, n_beta = build_model(path_to_model=PATH_TO_MODEL, model_name=MODEL_NAME)
    
    
    # Print losses
    print(f"""Final Loss: {tf.reduce_mean(build_loss(TRAINED_COEF_VALUES)
          (weights.astype(np.float32), np.concatenate((n_alpha(kinematics), n_beta(kinematics)), 1)))}""")
    
    
    # Plot the actual ratios versus estimated ratios for each comparison_coef, save if given a save_directory
    plot_ratios(quadratic=quadratic, 
                ratio_estimator=ratio_estimator, 
                n_alpha=n_alpha, 
                n_beta=n_beta, 
                kinematics=kinematics, 
                events=events, 
                comparison_coefs=COMPARISON_COEFS, 
                c_value_prefix=C_VALUE_PREFIX, 
                int_prefix=INT_PREFIX, 
                bsm_prefix=BSM_PREFIX, 
                save_directory=SAVE_DIRECTORY, 
                save_name=MODEL_NAME)
    
    


# Functions
def get_data(filepath: str, 
             interest_slice: slice, 
             coef_values: ty.Iterable, 
             randomize_data: bool, 
             c_value_prefix: str, 
             kinematic_columns: np.ndarray, 
             remove_negatives: bool, 
             use_weights: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # Load the data into a pandas DataFrame
    events = pd.read_pickle(filepath)

    # Properly normalize all the data
    nonnegative_array = events["weight_sm"] > 0.0
    for name in events.columns:
        if "weight_" in name:
            events[name] *= events["weight"]
            nonnegative_array &= events[name]
    
    # Remove the negatively-weighted events
    if remove_negatives:
        events = events[nonnegative_array]
    
    # Return the data
    training_data, weights = events_to_training(events=events[interest_slice], 
                                                coef_values=coef_values, 
                                                randomize_data=randomize_data, 
                                                c_value_prefix=c_value_prefix, 
                                                kinematic_columns=kinematic_columns)
    
    if use_weights:
        training_data = training_data / np.expand_dims(training_data[:, 0], -1)
    
    return events[interest_slice], training_data, weights


def build_model(path_to_model: str, model_name: str) -> tuple[bool, ty.Callable, Model | None, Model | None]:
    """Build the model

    Args:
        path_to_model (str): The path to the model
        model_name (str): The name of the model

    Returns:
        tuple[bool, ty.Callable]: Whether submodels were found, the model
    """
    
    # Get the paths of all files in the locations with the right name
    file_paths = glob(f"{path_to_model}/{model_name}*/")
    
    # Look for n_alpha and n_beta submodels
    found_submodels = False
    for path in file_paths:
        if "n_alpha" in path:
            n_alpha = load_model(path, compile=False)
            found_submodels = True
        if "n_beta" in path:
            n_beta = load_model(path, compile=False)
            found_submodels = True
    
    # If you found n_alpha and n_beta submodels, build the quadratic model out of them
    if found_submodels:
        def ratio_estimate(kinematics: np.ndarray, coefficents: np.ndarray) -> np.ndarray:
            return (1.0 + coefficents * n_alpha(kinematics)) ** 2.0 + (coefficents * n_beta(kinematics)) ** 2.0
        
        return True, ratio_estimate, n_alpha, n_beta
    # If you didn't find the submodels, assume it's the standard classifier and return that
    else:
        f_estimate = load_model(f"{path_to_model}/{model_name}", compile=False)
        def ratio_estimate(kinematics: np.ndarray) -> np.ndarray:
            return 1.0 / f_estimate(kinematics) - 1.0
        
        return False, ratio_estimate, None, None


def plot_ratios(quadratic: bool, 
                ratio_estimator: ty.Callable, 
                n_alpha: ty.Callable | None, 
                n_beta: ty.Callable | None, 
                kinematics: np.ndarray, 
                events: pd.DataFrame, 
                comparison_coefs: ty.Iterable, 
                c_value_prefix: str, 
                int_prefix: str, 
                bsm_prefix: str, 
                save_directory: str | None, 
                save_name: str) -> None:
    
    """Plot the estimated ratios compared to the true ones. 
    In addition, if the model is quadratic, plot the true linear and quadratic functions against
    the trained ones. 

    Args:
        quadratic (bool): Whether the model is quadratic or not
        ratio_estimator (ty.Callable): The function that estimates the cross-section ratio given the kinematics.
        n_alpha (ty.Callable | None): The alpha neural network, if the model is quadratic. None if not. 
        n_beta (ty.Callable | None): The beta neural network, if the model is quadratic. None if not.
        kinematics (np.ndarray): The kinematics of the events to plot
        events (pd.DataFrame): The events to plot
        comparison_coefs (ty.Iterable): The coefficients for which to plot the results
        c_value_prefix (str): The prefix of the c-value event columns
        int_prefix (str | None): The prefix of the interference weight columns
        bsm_prefix (str | None): The prefix of the beyond-standard-model weight columns
        save_directory (str | None): The directory in which to save the plots. No plots are saved if this is None
        save_name (str): The name with which to save the plots. 
    """
    
    for coef in comparison_coefs:
        # Plot the cross-section ratio
        actual_ratio = events[f"{c_value_prefix}{float_to_string(coef)}"] / events["weight_sm"]
        estimated_ratio = ratio_estimator(kinematics) if not quadratic else ratio_estimator(kinematics, coef)
        
        plt.axline((0.0, 0.0), slope=1.0, ls="--", c="r", zorder=0)
        plt.hist2d(np.squeeze(np.log(actual_ratio)), 
                   np.squeeze(np.log(estimated_ratio)), 
                   weights=np.abs(np.squeeze(events[f"{c_value_prefix}{float_to_string(coef)}"]) + np.squeeze(events["weight_sm"])), 
                   density=True,
                   bins=100)
        #                    norm=mpl.colors.LogNorm(), 
        # plt.scatter(np.log(actual_ratio), np.log(estimated_ratio), s=0.1)#, 
        #             c=np.abs(events[f"{c_value_prefix}{float_to_string(coef)}"] + events["weight_sm"]))
        # plt.colorbar()
        # plt.hist(np.log(actual_ratio), 
        #          weights=np.abs(events[f"{c_value_prefix}{float_to_string(coef)}"] + events["weight_sm"]), 
        #          density=True, 
        #          histtype="step", 
        #          bins=100, 
        #          ec="g")
        # plt.xlim(-1.0, 1.0)
        # plt.ylim(-1.0, 1.0)
        plt.title(f"Cross-Section Ratios for c = {coef} \n{np.mean(np.squeeze(np.abs(events[f'{c_value_prefix}{float_to_string(coef)}']) + np.squeeze(events['weight_sm'])) * (np.squeeze(np.log(estimated_ratio)) - np.squeeze(np.log(actual_ratio))) ** 2.0)}")
        plt.xlabel("True")
        plt.ylabel("Estimate")
        if save_directory is not None:
            plt.savefig(f"{save_directory}/{save_name}_ratios_{float_to_string(coef)}.png")
        # plt.show()
        
        # if quadratic:
        #     # Plot the linear part
        #     true_linear = events[f"{int_prefix}{float_to_string(coef)}"] / (2.0 * coef * events["weight_sm"])
        #     estimate_linear = n_alpha(kinematics)
            
        #     plt.axline((0.0, 0.0), slope=1.0, ls="--", c="r", zorder=0)
        #     plt.scatter(true_linear, estimate_linear, s=5.0)
        #     plt.title(rf"Actual vs. Trained $\alpha$ for c = {coef}")
        #     plt.xlabel("True")
        #     plt.ylabel("Estimate")
        #     if save_directory is not None:
        #         plt.savefig(f"{save_directory}/{save_name}_linear_{float_to_string(coef)}.png")
            
            # plt.show()
                
            # Plot the quadratic part
            # true_quadratic = np.sqrt(np.abs((events[f"{bsm_prefix}{float_to_string(coef)}"] 
            #                                  / ((coef ** 2.0) * events["weight_sm"])) - true_linear ** 2.0))
            # estimate_quadratic = np.abs(n_beta(kinematics))
            
            # plt.axline((0.0, 0.0), slope=1.0, ls="--", c="r", zorder=0)
            # plt.scatter(true_quadratic, estimate_quadratic, s=5.0)
            # plt.title(rf"Actual vs. Trained $\beta^2$ for c = {coef}")
            # plt.xlabel("True")
            # plt.ylabel("Estimate")
            # if save_directory is not None:
            #     plt.savefig(f"{save_directory}/{save_name}_quadratic_{float_to_string(coef)}.png")
            # plt.show()


if __name__ == "__main__":
    main()