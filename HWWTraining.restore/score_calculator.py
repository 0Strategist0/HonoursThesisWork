import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as npp
import matplotlib.pyplot as plt
from all_training_utils import float_to_string, string_to_float

print("Starting code")

# Format data and get c values
data = pd.read_pickle("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/shuffled_data.pkl")
c_values = []
weight_indices = []
for index, column in enumerate(data.columns):
    if "weight_" in column:
        data[column] *= data["weight"]
        if column != "weight_sm":
            c_values.append(string_to_float(column))
            weight_indices.append(index)
data = data.drop("weight", axis=1)
c_values = np.asarray(c_values)
weight_indices = np.asarray(weight_indices)

# Get the weights formatted for polynomial fitting
weight_ratios = (data.to_numpy() / data["weight_sm"].to_numpy()[:, np.newaxis])[:, weight_indices]
mod_weight_ratios = ((weight_ratios - 1.0) / c_values).T

# Get coefficients
coefficients = npp.polyfit(x=c_values, y=mod_weight_ratios, deg=1).T

# Define the augmented data
augmented_data = pd.DataFrame(np.concatenate((data.to_numpy(), coefficients), axis=1), columns=list(data.columns) + ["score", "quad_coef"])
print(np.shape(augmented_data))
print(augmented_data.columns)

pd.to_pickle(augmented_data, "augmented_data.pkl")


