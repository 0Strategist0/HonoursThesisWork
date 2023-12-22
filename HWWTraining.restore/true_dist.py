import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from all_training_utils import float_to_string, string_to_float

def main() -> None:
    
    print("Start")
    
    DATA_PATH = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/all_data.pkl"
    C_PREFIX = "weight_cHW_"
    C_VALUE = 0.05
    KINEMATIC_COLUMNS = np.arange(2, 39)
    
    data = pd.read_pickle(DATA_PATH)
    
    for name in data.columns:
        if "weight_" in name:
            data[name] *= data["weight"]
    
    
    weights = data[[f"{C_PREFIX}{float_to_string(c)}" for c in (-0.1, -0.05, -0.02, -0.01)] + ["weight_sm"] + [f"{C_PREFIX}{float_to_string(c)}" for c in (0.01, 0.02, 0.05, 0.1)]]
    
    N_sm = np.sum(data["weight_sm"].to_numpy())
    N_eft = np.sum(weights.to_numpy(), axis=0)
    
    plt.plot((-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1), N_eft)
    plt.savefig("N_eft.png")
    
    # Getting fake data
    fake_data_weights = (data[C_PREFIX + float_to_string(C_VALUE)] if C_VALUE != 0.0 else data["weight_sm"]).to_numpy()
    
    rng = np.random.default_rng(seed=122807528840384100672342137670123456789)
    n_samples = (rng.poisson(np.abs(fake_data_weights)) * np.sign(fake_data_weights)).astype(int)
    
    indices = np.concatenate([[index] * number for index, number in enumerate(np.abs(n_samples).astype(int))]).astype(int)
    
    fake_ldcsr = np.sum(np.sign(n_samples)[indices][:, np.newaxis] * np.log(weights.to_numpy() / weights["weight_sm"].to_numpy()[:, np.newaxis])[indices], axis=0)
    
    plt.figure()
    plt.plot((-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1), fake_ldcsr)
    plt.savefig(f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/True/sum_term{float_to_string(C_VALUE)}.png")
    
    print(fake_ldcsr)
    
    fake_data = np.concatenate((np.sign(n_samples)[indices][:, np.newaxis], data.to_numpy()[indices][:, KINEMATIC_COLUMNS]), axis=1)
    
    # Done
    
    print("Done faking data")
    
    ldcsr = np.sum(weights["weight_sm" if C_VALUE == 0.0 else C_PREFIX + float_to_string(C_VALUE)].to_numpy()[:, np.newaxis] * np.log(weights.to_numpy() / weights["weight_sm"].to_numpy()[:, np.newaxis]), axis=0)
    
    llr = -(N_eft - N_sm - ldcsr)
    lr = np.exp(llr)
    
    fake_llr = -(N_eft - N_sm - fake_ldcsr)
    fake_lr = np.exp(fake_llr)
    
    plt.figure()
    plt.plot((-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1), fake_llr)
    plt.savefig(f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/True/log_like_true_{float_to_string(C_VALUE)}.png")
    
    plt.figure()
    plt.plot((-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1), lr)
    plt.plot((-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1), fake_lr)
    plt.savefig(f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/True/true_dist_{float_to_string(C_VALUE)}.png")
    
    
    plt.figure()
    plt.plot((-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1), N_eft)
    plt.savefig(f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/True/NEFT_true_{float_to_string(C_VALUE)}.png")
    
    fake_data_pd = pd.DataFrame(fake_data)
    
    fake_data_pd.to_pickle(f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/TestingPlots/fake_inference{float_to_string(C_VALUE)}.pkl")


if __name__ == "__main__":
    main()