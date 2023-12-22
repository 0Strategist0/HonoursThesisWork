"""
# Analysis

---------------------------------------------------------------------------------------------------------------------------------

Kye Emond

August 21, 2023              
                                                                                                   
---------------------------------------------------------------------------------------------------------------------------------

A script to take a neural network, a test set of data, and a set of actual data, then use those to calculate a probability \
density for a Wilson Coefficient
"""

#### IMPORTS ####
import numpy as np
import numpy.typing as npt
import numpy.polynomial.polynomial as npp
import pandas as pd
import matplotlib.pyplot as plt
import keras as k
import typing as ty
import scipy.integrate as it
import scipy.special as sp
import scipy.optimize as op

from all_training_utils import string_to_float, float_to_string
from analysis_utils import weighted_mean, weighted_var, weighted_std

#### MAIN FUNCTION ####
C_VALUE = 0.05
def main() -> None:
    
    #### SETTINGS FOR THE ANALYSIS ####
    # Loading simulated events
    ALL_EVENTS_FILEPATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/AnalysisTest/full_data.pkl"
    TEST_FILEPATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/AnalysisTest/full_data.pkl"
    TEST_SLICE = slice(-10000, None)
    
    # Grabbing data from the loaded events
    KINEMATIC_COLUMNS = np.arange(2, 39)
    SM_TITLE = "weight_sm"
    C_VALUE_PREFIX = "weight_cHW_"
    FITTING_COEFS = (-0.5, -0.2, -0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
    
    # Actual data
    FAKE_DATA = True
    INFERENCE_FILEPATH = f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/fake_inference{float_to_string(C_VALUE)}.pkl"
    
    # Loading the neural network
    ALPHA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Results/MSEDirect_8_layers/cHW_MSEDirect_8_layers_n_alpha"
    BETA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Results/MSEDirect_8_layers/cHW_MSEDirect_8_layers_n_beta"
    
    # Decide what to display for the final plot
    CONFIDENCE_LEVEL = 0.7
    SAVE_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Results/c_pdf.png"
    
    
    
    #### ACTUAL ANALYSIS ####
    print("---- Loading Data ----")
    # Get the data used to generate the likelihood ratio estimator
    all_events = load_all_event_weights(filepath=ALL_EVENTS_FILEPATH, coefs=FITTING_COEFS, c_prefix=C_VALUE_PREFIX)
    test_kinematics, test_DCSR, test_weights = load_test_data(filepath=TEST_FILEPATH, 
                                                              kinematic_columns=KINEMATIC_COLUMNS, 
                                                              coefs=FITTING_COEFS, 
                                                              c_prefix=C_VALUE_PREFIX, 
                                                              sm_title=SM_TITLE,
                                                              test_slice=TEST_SLICE)
    
    # Get the data to do actual inference on
    inference_kinematics = load_inference_data(filepath=INFERENCE_FILEPATH, fake_data=FAKE_DATA)
    
    print("---- Building Estimator ----")
    
    # Build an estimator for the number of events
    num_events = build_event_number_estimator(all_events)
    
    # Get a function that gives the likelihood ratio along with bounds
    l_ratio_with_bounds = build_likelihood_ratio(alpha_path=ALPHA_PATH, 
                                                 beta_path=BETA_PATH, 
                                                 num_events=num_events, 
                                                 inference_data=inference_kinematics, 
                                                 test_inputs=test_kinematics, 
                                                 test_DCSR=test_DCSR, 
                                                 test_weights=test_weights, 
                                                 fake_data=FAKE_DATA)
    
    print("---- Testing Evaluation ----")
    
    print(l_ratio_with_bounds(np.linspace(-0.1, 0.1, 501), 0.7)[0])
    
    return
    
    print("---- Integrating Estimator ----")
    
    ############# THIS WILL NEED TO BE TESTED TO SEE WHERE LIKELIHOOD RATIO FAILS
    total_area = it.quad(lambda c: l_ratio_with_bounds(c, CONFIDENCE_LEVEL)[0], -np.inf, np.inf)
    
    print("---- Building Probability Density ----")
    
    # Define the probability density
    def pdf(c: npt.ArrayLike, confidence_level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ratio, lower_bound, upper_bound = l_ratio_with_bounds(c=c, confidence_level=confidence_level)
        
        return ratio / total_area, lower_bound / total_area, upper_bound / total_area
    
    print("---- Displaying Results ----")
    
    show_results(probability_density=pdf, confidence_level=CONFIDENCE_LEVEL, save_path=SAVE_PATH)
    




#### OTHER FUNCTIONS ####
def load_all_event_weights(filepath: str, coefs: ty.Iterable[float], c_prefix: str) -> pd.DataFrame:
    """Return a pandas DataFrame with the weights of each event for each Wilson Coefficient value. 

    Args:
        * filepath (str): A path to a pickle file containing the event weights in a pandas DataFrame.
        * coefs (ty.Iterable[float]): An iterable of Wilson Coefficients to load. 
        * c_prefix (str): The prefix of the weight columns in the pickle file's DataFrame.

    Returns:
        * pd.DataFrame: A pandas DataFrame containing event weights sorted by event along axis 0 and Wilson Coefficient value in \
            the columns. 
    """
    
    
    data = pd.read_pickle(filepath)
    columns = [f"{c_prefix}{float_to_string(coef)}" for coef in coefs]
    return data[columns]



def load_test_data(filepath: str, 
                   kinematic_columns: np.ndarray, 
                   coefs: ty.Iterable[float], 
                   c_prefix: str, 
                   sm_title: str, 
                   test_slice: slice | np.ndarray = slice(None, None)) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return a numpy array of kinematic variables for a bunch of test events, a pandas DataFrame of the differential cross-section \
        ratios for those same test events and various Wilson Coefficient values, and a pandas DataFrame of the weight of each event
        for those same Wilson Coefficient values. 

    Args:
        * filepath (str): A path to a pickle file holding a pandas DataFrame with all the event kinematics and weights. Should be \
            organized by event along axis 0 and category in the columns. 
        * kinematic_columns (np.ndarray): A numpy array of the column indices that contain kinematic variables.
        * coefs (ty.Iterable[float]): An iterable of Wilson Coefficient values to analyze. 
        * c_prefix (str): The prefix of the weight columns in the pickle file's DataFrame. 
        * sm_title (str): The column title for the standard model weights. 

    Returns:
        * np.ndarray: A numpy array containing the kinematic variables for all the events, organized with events along axis 0 \
            and variables along axis 1. 
        * pd.DataFrame: A pandas DataFrame holding differential cross-section ratios, with events along axis 0 and Wilson \
            Coefficient values categorized by column. 
        * pd.DataFrame: A pandas DataFrame holding the training weights for events, organized by event along axis 0 and \
            Wilson Coefficient values categorized by column. 
    """
    
    
    data = pd.read_pickle(filepath)[test_slice]
    columns = [f"{c_prefix}{float_to_string(coef)}" for coef in coefs]
    data_np = data.to_numpy()
    
    kinematics = data_np[:, kinematic_columns]
    
    dcsr = data[columns] * data[[sm_title]].to_numpy()
    
    test_weights = (data[columns] + data[[sm_title]].to_numpy()).abs()
    
    return kinematics, dcsr, test_weights



def load_inference_data(filepath: str, fake_data: bool = False) -> np.ndarray:
    """Return a numpy version of the pandas DataFrame stored in the pickle at filepath. 

    Args:
        * filepath (str): The path to the pickle file holding the DataFrame.
        * fake_data (bool): Whether the data was generated instead of experimental. Defaults to False.

    Returns:
        * (Optional) np.ndarray: An array with the number of times each event appears in the fake dataset.
        * np.ndarray: A numpy array of kinematic variables loaded from the pandas DataFrame.
    """
    
    data = pd.read_pickle(filepath).to_numpy()
    
    if not fake_data:
        return data
    else:
        return data[:, 0], data[:, 1:]



def build_event_number_estimator(full_dcs_dataset: pd.DataFrame, fit_degree: int = 2) -> ty.Callable[[npt.ArrayLike], float]:
    """Return a function to estimate the expected number of events at a given Wilson Coefficient value.

    Args:
        full_dcs_dataset (pd.DataFrame): A pandas dataframe with the weights for all the events along each column. 
        fit_degree (int, optional): The degree of the polynomial fit to the number of events. Defaults to 6.

    Returns:
        * ty.Callable: A function that returns an estimation of the expected number of events versus to the Wilson Coefficient \
            c.
            
            Args:
                * c (npt.ArrayLike): A 1D array of Wilson Coefficient values at which to evaluate the expected number of events. 

            Returns:
                * np.ndarray: An estimation of the number of events for each value of the Wilson Coefficient.
    """
    
    # Get the x and y values for the fit
    c_values = np.asarray([string_to_float(name) for name in full_dcs_dataset.columns])
    total_cross_sections = np.asarray(np.sum(full_dcs_dataset, axis=0))
    
    # Fit a function to the means vs c
    coefs = npp.polyfit(c_values, total_cross_sections, deg=fit_degree)
    def number_estimator(c: npt.ArrayLike) -> np.ndarray:
        return npp.polyval(c, coefs)
    
    
    return number_estimator
    
    

def build_likelihood_ratio(alpha_path: str, 
                           beta_path: str, 
                           num_events: ty.Callable[[npt.ArrayLike], np.ndarray], 
                           inference_data: npt.ArrayLike, 
                           test_inputs: npt.ArrayLike, 
                           test_DCSR: npt.ArrayLike, 
                           test_weights: npt.ArrayLike, 
                           fake_data: bool = False
                          ) -> ty.Callable[[npt.ArrayLike, float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return a function that gives the most probable likelihood ratio along with a confidence interval for an array of Wilson \
        Coefficients and a chosen confidence level.

    Args:
        * alpha_path (str): The path to the saved alpha neural network.
        * beta_path (str): The path to the saved beta neural network. 
        * num_events (ty.Callable): A callable that returns the expected number of total events for a given value of the Wilson \
            Coefficient. 
        * inference_data (npt.ArrayLike): A 2D array_like of observed kinematic variables. Individual events should be arranged \
            along axis 0 and different variables should be arranged along axis 1.
        * test_inputs (npt.ArrayLike): The input kinematics for a set of test data used to estimate network accuracy. 
        * test_DCSR (npt.ArrayLike): The true differential cross section ratios for all the test events. 
        * test_weights (npt.ArrayLike): The event weight for each of the test events.
        * fake_data (bool): Whether the data is fake (generated) or not. Defaults to False.
    
    Returns:
        * ty.Callable: A function that returns the most probable likelihood ratio along with a confidence interval for an array \
            of Wilson Coefficients and a chosen confidence interval
            
            Args:
                * c (npt.ArrayLike): A 1D array of Wilson Coefficient values at which to evaluate the likelihood ratio and its \
                    uncertainty.
                * confidence_level (float): The confidence level for which to determine the confidence interval.

            Returns:
                * np.ndarray: A 1D array of the most probable values for the likelihood ratios at the given Wilson Coefficient \
                    values.
                * np.ndarray: A 1D array of the lower bounds on the confidence intervals for each of the given Wilson \
                    Coefficient values. 
                * np.ndarray: A 1D array of the upper bounds on the confidence intervals for each of the given Wilson \
                    Coefficient values. 
    """
    
    # Build differential cross-section (DCS) ratio estimator 
    n_alpha = k.models.load_model(alpha_path, compile=False)
    n_beta = k.models.load_model(beta_path, compile=False)
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
    
    # Get the sample mean and variance as functions of c based on the test set
    # total_mean, total_variance = get_mean_and_variance(dcs_estimator=best_dcross_section_ratio, 
    #                                                          test_inputs=test_inputs, 
    #                                                          test_outputs=test_DCSR, 
    #                                                          weights=test_weights)
    
    # Use DCS estimator to get the log likelihood ratio (lLR) estimator
    if not fake_data:
        def log_l_ratio(c: npt.ArrayLike, data: npt.ArrayLike) -> np.ndarray:
            """Return the total log likelihood ratio (LLR) estimates by the neural network, including corrections for mean residuals.

            Args:
                * c (npt.ArrayLike): A 1D array_like that contains the values of the Wilson Coefficient for which to evaluate LLRs.
                * data (npt.ArrayLike): A 2D array_like of kinematic variables. Individual events should be arranged along axis 0 and \
                    different variables should be arranged along axis 1.

            Returns:
                * np.ndarray: A 1D array of the estimated LLR for each element in `c`.
            """
            
            network_estimate = num_events(c) - num_events(0) - np.sum(np.log(best_dcross_section_ratio(c=c, data=data)), axis=0)
            correction = -total_mean(c, data.shape[0])
            
            return network_estimate + correction
    else:
        def log_l_ratio(c: npt.ArrayLike, data: npt.ArrayLike) -> np.ndarray:
            """Return the total log likelihood ratio (LLR) estimates by the neural network, including corrections for mean residuals.

            Args:
                * c (npt.ArrayLike): A 1D array_like that contains the values of the Wilson Coefficient for which to evaluate LLRs.
                * data (npt.ArrayLike): A 2D array_like of kinematic variables. Individual events should be arranged along axis 0 and \
                    different variables should be arranged along axis 1.

            Returns:
                * np.ndarray: A 1D array of the estimated LLR for each element in `c`.
            """
            
            print("Calculating Log Likelihood Ratio")
            
            print("Event number difference:", num_events(c) - num_events(0))
            print("Differential Cross-Section Difference", np.sum(data[0][:, np.newaxis]
                                                                      * np.log(best_dcross_section_ratio(c=c, data=data[1])), 
                                                                      axis=0))
            # print("Mean correction", total_mean(c, np.sum(np.abs(data[0]))))
            
            print(np.shape(np.sum(data[0][:, np.newaxis]
                                                                      * np.log(best_dcross_section_ratio(c=c, data=data[1])), 
                                                                      axis=0)))
            plt.figure()
            plt.plot(c, (np.sum(data[0][:, np.newaxis]
                                * np.log(best_dcross_section_ratio(c=c, data=data[1])), 
                                axis=0) + len(data[0]) * (-2.70125594e-8 - 4.50375350e-4 * c - 1.50615688e-2 * c ** 2.0)))
            plt.savefig(f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/Fake/sum_term_approx{float_to_string(C_VALUE)}.png")
            print("ALSO DONE")
            
            network_estimate = num_events(c) - num_events(0) - (np.sum(data[0][:, np.newaxis]
                                                                * np.log(best_dcross_section_ratio(c=c, data=data[1])), 
                                                                axis=0) + len(data[0]) * (-2.70125594e-8 - 4.50375350e-4 * c - 1.50615688e-2 * c ** 2.0))
            
            plt.figure()
            plt.plot(c, num_events(c))
            plt.savefig(f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/Fake/NUMEFT_fake{float_to_string(C_VALUE)}.png")
            
            plt.figure()
            plt.plot(c, -network_estimate)
            plt.savefig(f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/Fake/neg_log_like_approx{float_to_string(C_VALUE)}.png")
            
            plt.figure()
            plt.plot(c, np.exp(-network_estimate))
            plt.axvline(C_VALUE)
            plt.savefig(f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/Fake/please_work_approx{float_to_string(C_VALUE)}.png")
            # correction = -total_mean(c, np.sum(np.abs(data[0])))
            
            # print(network_estimate + correction)
            
            #TODO fix correction term
            
            return network_estimate# + correction
    
    # Use the lLR estimator and mean(c) to get a likelihood ratio (LR) estimator (flipping the ratio though
    # so SM is in the denominator and EFT is in the numerator)
    def l_ratio(c: npt.ArrayLike, data: npt.ArrayLike) -> np.ndarray:
        """Return the most probable likelihood ratios (LR) based on the neural network's characteristics.

        Args:
            * c (npt.ArrayLike): A 1D array_like that contains the values of the Wilson Coefficient for which to evaluate LRs.
            * data (npt.ArrayLike): A 2D array_like of kinematic variables. Individual events should be arranged along axis 0 and \
                different variables should be arranged along axis 1.

        Returns:
            * np.ndarray: A 1D array of the estimated LR for each element in `c`.
        """
        
        print("Calculating actual likelihood ratio")
        
        # print("Without Variance", np.exp(-(log_l_ratio(c=c, data=data))))
        y = np.exp(-(log_l_ratio(c=np.linspace(-0.1, 0.1, 501), data=data)))
        plt.figure()
        plt.clf()
        plt.plot(np.linspace(-0.1, 0.1, 501), y)
        plt.savefig(f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/Fake/test{float_to_string(C_VALUE)}.png")
        print(y.shape)
        exit()
        
        # print(total_variance(c, data.shape[0] if not fake_data 
        #                                                              else np.sum(np.abs(data[0]))))
        
        return np.exp(-(log_l_ratio(c=c, data=data) + total_variance(c, data.shape[0] if not fake_data 
                                                                     else np.sum(np.abs(data[0])))))
        
    
    # Get the CDF and inverse CDF of the likelihood ratio
    def cdf_l_ratio(lr: npt.ArrayLike, c: npt.ArrayLike, data: npt.ArrayLike) -> np.ndarray:
        """Return the probability that the true value of the likelihood ratio (LR) is less than `lr` for given Wilson \
            Coefficients and observed data.

        Args:
            * lr (npt.ArrayLike): A 1D array_like containing the LRs at which to evaluate the CDF. 
            * c (npt.ArrayLike): A 1D array_like containing the Wilson Coefficient values to use when parametrizing the CDF for \
                each LR. The size of `c` must match the size of `lr`.
            * data (npt.ArrayLike): A 2D array_like of kinematic variables. Individual events should be arranged along axis 0 and \
                different variables should be arranged along axis 1.

        Returns:
            * np.ndarray: A 1D array of the probability for each LR and Wilson Coefficient value pair. 
        """
        
        print("calculating lr CDF")
        
        better_lr = np.ravel(lr)
        better_c = np.ravel(c)
        
        return 1.0 - sp.ndtr((-np.log(better_lr) - log_l_ratio(c=better_c, data=data)) 
                             / total_variance(c, data.shape[0] if not fake_data else np.sum(np.abs(data[0]))))
    
    
    def inv_cdf_l_ratio(prob: npt.ArrayLike, c: npt.ArrayLike, data: npt.ArrayLike) -> np.ndarray:
        """Return the likelihood ratio (LR) quantile corresponding to `prob`.

        Args:
            * prob (npt.ArrayLike): A 1D array_like containing the probabilities at which to find the LRs. 
            * c (npt.ArrayLike): A 1D array_like containing the Wilson Coefficient values to use when parametrizing this function \
                for each probability. The size of `c` must match the size of `prob`.
            * data (npt.ArrayLike): A 2D array_like of kinematic variables. Individual events should be arranged along axis 0 and \
                different variables should be arranged along axis 1.

        Returns:
            * np.ndarray: A 1D array of the quantiles for each probability and Wilson Coefficient value pair. 
        """
        
        print("Calculating inverse lr CDF")
        
        better_prob = np.ravel(prob)
        better_c = np.ravel(c)
        
        return np.exp(-np.sqrt(total_variance(c, data.shape[0] if not fake_data else np.sum(np.abs(data[0])))) 
                      * sp.ndtri(1.0 - better_prob) - log_l_ratio(c=better_c, data=data))
    
    
    # Build upper/lower bound estimator for a given confidence level
    def bounds(confidence_level: float, c: npt.ArrayLike, data: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the upper and lower bounds for a minimal confidence interval containing the most probable value (MPV) for the \
            true likelihood ratio (LR) at a given `confidence_level`.

        Args:
            * confidence_level (float): The confidence level for which to calculate the minimal confidence interval. Must be in \
                the range (0, 1).
            * c (npt.ArrayLike): A 1D array_like containing the Wilson Coefficient values for which to calculate the confidence \
                intervals.
            * data (npt.ArrayLike): A 2D array_like of kinematic variables. Individual events should be arranged along axis 0 and \
                different variables should be arranged along axis 1.

        Returns:
            * np.ndarray: A 1D array containing the lower bounds corresponding to each Wilson Coefficient value.
            * np.ndarray: A 1D array containing the upper bounds corresponding to each Wilson Coefficient value.
        """
        
        new_c = np.reshape(c, np.asarray(c).size)
        
        def summed_interval_sizes(start: npt.ArrayLike) -> float:
            return np.sum(inv_cdf_l_ratio(prob=confidence_level + cdf_l_ratio(lr=start, c=new_c, data=data), 
                                          c=new_c, 
                                          data=data) 
                          - start)
        
        
        # Find the lower bounds that minimize the confidence interval sizes
        optimization_object = op.minimize(summed_interval_sizes, 
                                          l_ratio(c=new_c, data=data) / 2.0, 
                                          bounds=np.array(np.zeros(len(new_c)), l_ratio(c=new_c, data=data)).T)
        
        if not optimization_object.success:
            print("ERROR: Failed to find minimal confidence intervals")
            print(optimization_object.message)
        
        best_starts = optimization_object.x
        
        # Use those lower bounds to calculate the upper bounds
        best_ends = inv_cdf_l_ratio(prob=confidence_level + cdf_l_ratio(lr=best_starts, c=new_c, data=data), 
                                    c=new_c, 
                                    data=data)
        
        # Return the bounds
        return best_starts, best_ends
    
    
    # Define the function to return the likelihood ratio with the appropriate bounds
    def l_ratio_with_bounds(c: npt.ArrayLike, confidence_level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the most probable likelihood ratio (LR) and the bounds of a confidence interval at a given confidence \
            level for and array of Wilson Coefficient values. 

        Args:
            * c (npt.ArrayLike): A 1D array of Wilson Coefficient values at which to evaluate the likelihood ratio and its \
                uncertainty.
            * confidence_level (float): The confidence level for which to determine the confidence interval.

        Returns:
            * np.ndarray: A 1D array of the most probable values for the likelihood ratios at the given Wilson Coefficient \
                values.
            * np.ndarray: A 1D array of the lower bounds on the confidence intervals for each of the given Wilson \
                Coefficient values. 
            * np.ndarray: A 1D array of the upper bounds on the confidence intervals for each of the given Wilson \
                Coefficient values. 
        """
        # TODO: Fix this
        
        print("Figuring out likelihood ratio with bounds")
        
        return (l_ratio(c=c, data=inference_data),) * 3 #*bounds(confidence_level=confidence_level, c=c, data=inference_data))
        
    return l_ratio_with_bounds



def get_mean_and_variance(dcs_estimator: ty.Callable[[npt.ArrayLike, npt.ArrayLike], np.ndarray], 
                          test_inputs: npt.ArrayLike, 
                          test_outputs: pd.DataFrame, 
                          weights: npt.ArrayLike, 
                          mean_fit_degree: int = 5, 
                          var_fit_degree: int = 6
                          ) -> tuple[ty.Callable[[npt.ArrayLike, float], np.ndarray], 
                                     ty.Callable[[npt.ArrayLike, float], np.ndarray]]:
    """Return two functions to esimate the mean and variance of the neural network residuals for a certain value of a Wilson \
        Coefficient

    Args:
        * dcs_estimator (ty.Callable[[npt.ArrayLike, npt.ArrayLike], np.ndarray]): A function that takes in an array of Wilson \
            Coefficient values and an array of kinematic events with events along axis 0 and variables along axis 1, then \
            returns an array of estimated differential cross-section ratios, with events along axis 0 and coefficients along \
            axis 1. 
        * test_inputs (npt.ArrayLike): The set of kinematic variables to use for testing the dcs_estimator. Must have events \
            along axis 0 and variables along axis 1. 
        * test_outputs (pd.DataFrame): A pandas DataFrame holding the differential cross-sections for each test event and Wilson \
            Coefficient value, with column titles labelled in a way that can be converted to a float using string_to_float. 
        * weights (npt.ArrayLike): A 2D array of event weights, with events along axis 0 and Wilson Coefficient value along axis 1.
        * mean_fit_degree (int, optional): The degree of the polynomial to fit the mean function. Defaults to 5.
        * var_fit_degree (int, optional): The degree of the polynomial to fit the variance function. Defaults to 6.

    Returns:
        * ty.Callable: A function that returns an estimation of the mean offset of the neural network from the true value.
            
            Args:
                * c (npt.ArrayLike): A 1D array of Wilson Coefficient values at which to evaluate the mean. 

            Returns:
                * np.ndarray: An estimation of the mean offset of the neural network from the true value.
        * ty.Callable: A function that returns an estimation of the variance of the neural network residuals.
            
            Args:
                * c (npt.ArrayLike): A 1D array of Wilson Coefficient values at which to evaluate the variance. 

            Returns:
                * np.ndarray: An estimation of the variance in the neural network residuals.
    """
    
    
    # Get values of c
    c_values = [string_to_float(name) for name in test_outputs.columns]
    
    # Evaluate dcs on test_inputs
    output_estimates = dcs_estimator(c=c_values, data=test_inputs)
    
    # Get test outputs as a numpy array
    test_outputs_np = test_outputs.to_numpy()
    
    # Get residuals of log(dcs estimate) to true outputs
    residuals = np.log(test_outputs_np) - np.log(output_estimates)
    
    # Get the means of the residuals vs c
    means = weighted_mean(residuals, weights=weights, axis=0)
    
    # Get the variances of the residuals vs c
    vars = weighted_var(residuals, weights=weights, axis=0)
    
    # Get the uncertainty in the means of the residuals vs c
    mean_uncert = weighted_std(residuals, weights=weights, axis=0) / np.sqrt(residuals.shape[0])
    
    # Get the approximate uncertainty in the variances of the residuals vs c
    var_uncert = vars * np.sqrt(2.0 / (residuals.shape[0] - 1.0))
    
    # Fit a function to the means vs c
    mean_coefs = npp.polyfit(c_values, means, deg=mean_fit_degree, w=1.0 / mean_uncert)
    def mean_func(c: npt.ArrayLike, num: float) -> np.ndarray:
        return npp.polyval(c, mean_coefs) * num
    
    # Fit a function to the variances vs c
    var_coefs = npp.polyfit(c_values, vars, deg=var_fit_degree, w=1.0 / var_uncert)
    def var_func(c: npt.ArrayLike, num: float) -> np.ndarray:
        return npp.polyval(c, var_coefs) * num
    
    # Return the mean and variance functions
    return mean_func, var_func



def show_results(probability_density: ty.Callable[[npt.ArrayLike], tuple[np.ndarray, np.ndarray, np.ndarray]], 
                 confidence_level: float, 
                 save_path: str, 
                 xlimits: tuple[float, float] = (-0.5, 0.5), 
                 ylimits: tuple[float, float] | None = None, 
                 xlabel: str = "Wilson Coefficient Value", 
                 ylabel: str = "Probability Density", 
                 title: str = "") -> None:
    """Generate a plot of the probability density for the Wilson Coefficient and its confidence interval. 

    Args:
        * probability_density (ty.Callable[[npt.ArrayLike], tuple[np.ndarray, np.ndarray, np.ndarray]]): A function which takes \
            in an array_like of Wilson Coefficient values and returns the best estimate of the probability density, along with \
            upper and lower bounds for that probability density.
        * confidence_level (float): The confidence level to use when generating the bounds on the probability density. 
        * save_path (str): Where to save the generated plot. 
        * xlimits (tuple[float, float], optional): The x limits of the plot. Defaults to (-0.5, 0.5).
        * ylimits (tuple[float, float] | None, optional): The y limits of the plot. Defaults to None.
        * xlabel (str, optional): The label for the x axis. Defaults to "Wilson Coefficient Value".
        * ylabel (str, optional): The label for the y axis. Defaults to "Probability Density".
        * title (str, optional): The title of the plot. Defaults to "".
    """
    
    plt.figure(constrained_layout=True)
    
    # Calculate the probability density and its uncertainties
    c_values = np.linspace(*xlimits, 100)
    best_pdf, lower_pdf, upper_pdf = probability_density(c=c_values, confidence_level=confidence_level)
    
    # Plot those
    plt.fill_between(c_values, upper_pdf, lower_pdf, alpha=0.5)
    plt.plot(c_values, best_pdf)
    
    # Formatting
    plt.xlim(xlimits)
    if ylimits is not None:
        plt.ylim(ylimits)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Save
    plt.savefig(save_path)



#### RUN SCRIPT ####
if __name__ == "__main__":
    main()