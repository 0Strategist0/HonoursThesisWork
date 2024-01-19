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
import matplotlib.pyplot as plt
import typing as ty
import numpy.typing as npt

# Main function
def main() -> None:
    
    # Loading data
    TEST_DATA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/Data/AnalysisTest/fake_inference0.pkl"
    MODEL_DIRECTORY = "/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/Results/FullScoreTest"
    
    # Load and format the test data
    with open(TEST_DATA_PATH, "rb") as datafile:
        data = pkl.load(datafile)
        print(type(data))
    
    # Load all the models from the directory into a dictionary
    
    # Iteratively evaluate each model on the test data and store its performance metrics and some plots
    
    # Display the performance metrics for each model and make some plots to compare them
    
    


# Call the main function
if __name__ == "__main__":
    main()