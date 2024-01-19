"""
# Generate Test Data

---------------------------------------------------------------------------------------------------------------------------------

Kye Emond

January 11, 2024

---------------------------------------------------------------------------------------------------------------------------------

A script designed to use a set of events and their weights for a variety of Wilson Coefficient values to generate a realistic \
dataset, along with a test dataset storing all the actual information about those events.
"""

import numpy as np
import pandas as pd
import pickle as pkl

# from all_training_utils import float_to_string

def main() -> None:
    
    # Constants for reading data
    DATA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/Data/shuffled_data.pkl"
    C_PREFIX = "weight_cHW_"
    C_VALUE = 0.05
    KINEMATIC_COLUMNS = np.arange(2, 39)
    WEIGHT_COLUMN_NAMES_TO_SAVE = ['weight_sm', 'weight_cHW_pos0p01', 'weight_cHW_pos0p02', 'weight_cHW_pos0p05', 
                                   'weight_cHW_pos0p1', 'weight_cHW_pos0p2', 'weight_cHW_pos0p5', 'weight_cHW_pos1p0', 
                                   'weight_cHW_pos2p0', 'weight_cHW_neg0p01', 'weight_cHW_neg0p02', 'weight_cHW_neg0p05', 
                                   'weight_cHW_neg0p1', 'weight_cHW_neg0p2', 'weight_cHW_neg0p5', 'weight_cHW_neg1p0',
                                   'weight_cHW_neg2p0']
    
    # Randomization
    SEED = 122807528840384100672342137670123435476
    
    # Saving
    FAKE_DATA_SAVE_PATH = f"/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/TestingModels/FakeData/fake_data_c={C_VALUE}.pkl"
    FAKE_WEIGHTS_SAVE_PATH = f"/home/kye/projects/ctb-stelzer/kye/HWWTraining.restore/TestingModels/FakeData/fake_weights_c={C_VALUE}.pkl"
    
    
    
    # Load the set of events and weights
    data = pd.read_pickle(DATA_PATH)
    
    
    # Iterate through the events and weights to get corrected weights
    for name in data.columns:
        if "weight_" in name:
            data[name] *= data["weight"]
    
    # Get the array of weights for each event at the given Wilson Coefficient value
    weights = (data[C_PREFIX + float_to_string(C_VALUE)] if C_VALUE != 0.0 else data["weight_sm"]).to_numpy()
    
    
    
    # Randomize
    rng = np.random.default_rng(seed=SEED)
    n_samples = (rng.poisson(np.abs(weights)) * np.sign(weights)).astype(int)
    
    # Get an array of the indices to use when picking events. For example, if the 0th event is picked once, the 
    # 1st event is picked twice, and the 2nd even is picked twice, it would generate [0, 1, 1, 2, 2]
    indices = np.concatenate([[index] * number for index, number in enumerate(np.abs(n_samples).astype(int))]).astype(int)
    
    # Make a set of data with a column indicating whether the events are positively or negatively weighted, followed
    # by the events
    fake_data = pd.DataFrame(
        np.concatenate((np.sign(n_samples)[indices][:, np.newaxis], data.to_numpy()[indices][:, KINEMATIC_COLUMNS]), axis=1), 
        columns=["sign"] + list(data.columns[KINEMATIC_COLUMNS]))
    
    # Save it
    fake_data.to_pickle(FAKE_DATA_SAVE_PATH)
    
    # Make a set of data giving the weights for each event that was put in the fake data
    fake_weights = pd.DataFrame(
        data[WEIGHT_COLUMN_NAMES_TO_SAVE].to_numpy()[indices], 
        columns = WEIGHT_COLUMN_NAMES_TO_SAVE
    )
    
    # Save it
    fake_weights.to_pickle(FAKE_WEIGHTS_SAVE_PATH)


def float_to_string(number: float) -> str:
    """Return the weird nTuple-formatted string version of a float.

    Args:
        number (float): The number to convert.

    Returns:
        str: The weird formatted string version of the float.
    """
    
    string = str(number).replace(".", "p").replace("-", "neg")
    if "neg" not in string:
        string = "pos" + string
    return string


if __name__ == "__main__":
    main()