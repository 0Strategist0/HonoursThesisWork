import numpy as np
import numpy.polynomial.polynomial as npp
import pandas as pd
import matplotlib.pyplot as plt
from all_training_utils import float_to_string, string_to_float


all_data = pd.read_pickle("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/augmented_data.pkl")
print(np.count_nonzero(~np.isfinite(all_data)))
# print(np.array([all_data["weight_cHW_" + float_to_string(value)] for value in [1.0, -1.0]]).T.shape)

# c_values = [-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1]

# print(np.shape(all_data[("weight_cHW_" + float_to_string(0.02)) if 0.02 != 0.0 else "weight_sm"].to_numpy()))
# print(np.shape(all_data[("weight_cHW_" + float_to_string(0.02)) if 0.02 != 0.0 else "weight_sm"].to_numpy()[10]))

# # print(np.shape([np.squeeze(all_data["weight_cHW_" + float_to_string(c) if c != 0.0 else "weight_sm"][10]) / np.squeeze(all_data["weight_sm"][10]) for c in c_values]))

# plt.scatter(c_values, [np.squeeze(all_data["weight_cHW_" + float_to_string(c) if c != 0.0 else "weight_sm"].to_numpy()[10]) / np.squeeze(all_data["weight_sm"].to_numpy()[10]) for c in c_values])
# coefs = npp.polyfit(c_values, [np.squeeze(all_data["weight_cHW_" + float_to_string(c) if c != 0.0 else "weight_sm"].to_numpy()[10]) / np.squeeze(all_data["weight_sm"].to_numpy()[10]) for c in c_values], deg=2)
# def mean_func(c) -> np.ndarray:
#     return npp.polyval(c, coefs)

# plt.plot(np.linspace(-0.1, 0.1, 100), mean_func(np.linspace(-0.1, 0.1, 100)))
# plt.xlabel("c")
# plt.ylabel("$d\sigma_{SM}/d\sigma_{EFT}(c)$")
# plt.savefig("Quadratic.png")


# all_data = all_data.sample(frac=1)

# print(all_data.shape)
# print(all_data.columns)

# all_data.to_pickle("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/shuffled_data.pkl")