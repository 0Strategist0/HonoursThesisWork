"""
# All Training Utils

---------------------------------------------------------------------------------------------------------------------------------

Kye Emond

June 15, 2023

---------------------------------------------------------------------------------------------------------------------------------

A module designed to help training a neural network for cross-section ratio estimation, using all events to their maximum \
potential
"""

print("Code running")

# Imports
import sys
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
import typing as ty
import numpy.typing as npt

# A module I wrote for this specific neural network project. Contains several loss functions as well as some functions to make
# setting up data and the network itself easier. 
from all_training_utils import build_score_loss, simple_deep_dense_net, float_to_string, EpochModelCheckpoint


# Main function
def main() -> None:
    
    print("\n" + "START".center(50, "-"))
    
    # # Try to avoid memory fragmentation?
    # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    # print(os.getenv("TF_GPU_ALLOCATOR"))
    
    # Get the arguments for the file
    (job_number,) = get_args(("jobnumber",))
    job_number = int(job_number) if job_number is not None else None
    
    # Loading data
    PATH_TO_DATA = "/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/augmented_data.pkl"
    C_VALUE_PREFIX = "weight_cHW_"
    KINEMATIC_COLUMNS = np.arange(2, 39)
    RANDOMIZE_DATA = True
    
    # What data to use for training
    TRAINING_SLICE = slice(None, -1_000_000)
    BATCH_SIZE = int(2 ** 20)
    COEF_VALUES = (-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    
    # Network configuration
    HIDDEN_LAYERS = (30, 30, 30)
    DROPOUT_FRAC = 0.0
    BATCH_NORMALIZATION = False
    
    # Training settings
    OPTIMIZER = k.optimizers.Adam(learning_rate=0.0001, clipnorm=0.01)#k.optimizers.SGD(momentum=0.9, nesterov=True, learning_rate=0.000001)
    EPOCHS = 500_000
    PATIENCE = 15
    MIN_DELTA = 0.0
    REDUCE_LR_ON_PLATEAU = False
    VALIDATION_SPLIT = 0.2
    MSE_FRAC = (job_number % 5) / (5.0 - 1.0)
    SCORE_SUBFRAC = (job_number // 5) / (5.0 - 1.0)
    
    # Checkpointing
    CHECKPOINT_PATH = None #f"/project/6024950/kye/HWWTraining/Checkpoints/{LOSS_TYPE}"
    CHECKPOINT_PERIOD = 100
    
    # Saving
    SAVE_DIRECTORY = (f"/home/kye/projects/ctb-stelzer/kye/HWWTraining/Results/FullScoreTest/MSEFRAC{float_to_string(MSE_FRAC)}_SCORESUBFRAC{float_to_string(SCORE_SUBFRAC)}")
    NETWORK_NAME = f"cHW_score_test_30x30x30"
    
    
    # Make directory for saving
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
    
    print("\n" + "SETTING UP MIRRORED STRATEGY".center(50, "-"))
    
    # Set up the training on as many GPUs as available
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
    # Start defining variables on all those GPUs
    with strategy.scope():
        print("\n" + "LOADING DATA".center(50, "-"))
        
        # Get the training data
        training_data, true = get_data(filepath=PATH_TO_DATA, 
                                       training_slice=TRAINING_SLICE, 
                                       coef_values=COEF_VALUES, 
                                       c_value_prefix=C_VALUE_PREFIX, 
                                       kinematic_columns=KINEMATIC_COLUMNS, 
                                       randomize_data=RANDOMIZE_DATA)
        

        print(f"Training Data Shape: {training_data.shape}, Memory: {training_data.nbytes}")
        print(f"True Data Shape: {true.shape}, Memory: {true.nbytes}")
        
        print("\n" + "BUILDING MODEL".center(50, "-"))
        
        # Build the model
        model = build_model(training_data=training_data, 
                            hidden_layers=HIDDEN_LAYERS, 
                            dropout_frac=DROPOUT_FRAC, 
                            optimizer=OPTIMIZER, 
                            training_coefs=COEF_VALUES, 
                            MSE_frac=MSE_FRAC, 
                            score_subfrac=SCORE_SUBFRAC, 
                            batch_normalization=BATCH_NORMALIZATION) 
        
    print("\n" + "TRAINING MODEL".center(50, "-"))
    
    # Train the model
    history = train_model(model=model, 
                          training_data=training_data, 
                          true=true, 
                          batch_size=BATCH_SIZE, 
                          epochs=EPOCHS, 
                          patience=PATIENCE, 
                          min_delta=MIN_DELTA, 
                          reduce_lr_on_plateau=REDUCE_LR_ON_PLATEAU, 
                          checkpoint_path=CHECKPOINT_PATH, 
                          checkpoint_frequency=CHECKPOINT_PERIOD, 
                          validation_split=VALIDATION_SPLIT)
    
    print("\n" + "SAVING".center(50, "-"))
    
    # Save everything
    save_run(model=model, history=history, network_name=NETWORK_NAME, save_directory=SAVE_DIRECTORY)
    
    print("\n" + "GENERATING PLOTS".center(50, "-"))
    
    # Set up some plots and give root mean squared error/root mean squared log error for comparison with other models
    make_comparisons(n_alpha=extract_submodel(model, ("input_1", "normalization", "n_alpha")), 
                     n_beta=extract_submodel(model, ("input_1", "normalization", "n_beta")), 
                     coefficients=COEF_VALUES, 
                     val_input=training_data[-int(len(training_data) * VALIDATION_SPLIT):], 
                     val_ratios=((true[:, 1:-2] / true[:, 0:1])[-int(len(training_data) * VALIDATION_SPLIT):]).T, 
                     event_weights=np.abs(true[:, 1:-2] + true[:, 0:1])[-int(len(training_data) * VALIDATION_SPLIT):].T, 
                     network_name=NETWORK_NAME, 
                     save_directory=SAVE_DIRECTORY)


# Functions used in main
def get_data(filepath: str, 
             training_slice: slice, 
             coef_values: ty.Iterable, 
             c_value_prefix: str, 
             kinematic_columns: npt.ArrayLike, 
             randomize_data: bool, 
             SM_weight_title: str = "weight_sm") -> tuple[np.ndarray, np.ndarray]:
    """Return properly formatted data for training

    Args:
        * filepath (str): Path to the root nTuples
        * training_slice (slice): The slice of events to use for training
        * coef_values (Iterable): The values of the coefficients to train on
        * c_value_prefix (str): The prefix of the coefficient values in the root nTuple columns
        * kinematic_columns (ndarray): A numpy array of indices where you can find the kinematic variables
        * randomize_data (bool): Whether to randomize the output data
        * SM_weight_title (str): The title of the Standard Model weight column. Defaults to "weight_sm".

    Returns:
        * ndarray: The input variables for training
        * ndarray: The weights for the loss function
    """
    
    # Load the data into a pandas DataFrame
    events = pd.read_pickle(filepath)[training_slice]
    
    # Convert the events to numpy
    events_np = events.to_numpy()
    
    # Get the training data
    training_data = events_np[:, kinematic_columns]
    
    # Get the true values to test network against
    sm_weights = events[[SM_weight_title]].to_numpy()
    weights = np.array([events[c_value_prefix + float_to_string(value)] for value in coef_values]).T
    coefs = events_np[:, -2:]
    
    true = np.concatenate((sm_weights, weights, coefs), axis=1)
    
    # Randomize the data
    shuffled_indices = np.random.permutation(len(training_data)) if randomize_data else np.arange(len(training_data))
    
    return training_data[shuffled_indices], true[shuffled_indices]


def get_args(names: ty.Iterable[str]) -> tuple[str, ...]:
    
    # Get the set of arguments passed when running the script
    arguments = sys.argv
    
    # Set up the list to return
    returns = [None for _ in names]
    
    # Iterate through and check if there was a filepath
    for arg in arguments:
        for index, name in enumerate(names):
            if f"--{name}=" in arg or f"-{name[0]}=" in arg:
                returns[index] = arg.split("=")[1]
    
    # Return the arguments
    return returns


def build_model(training_data: tf.Tensor, 
                hidden_layers: int, 
                dropout_frac: float, 
                optimizer: str | k.optimizers.Optimizer, 
                training_coefs: ty.Iterable,
                MSE_frac: float, 
                score_subfrac: float, 
                batch_normalization: bool) -> k.models.Model:
    """Returns the constructed DNN

    Args:
        * training_data (Tensor): The input variables to train on
        * hidden_layers (int): The number of hidden layers to use
        * dropout_frac (float): The fraction for the dropout layers
        * optimizer (str | k.optimizers.Optimizer): The optimizer to use for the training
        * training_coefs (Iterable): The values of the coefficients to train on
        * MSE_frac (float): The fraction of the loss attributed to mean squared error
        * score_subfrac (float): The fraction of the loss NOT attributed to mean squared error that should be attributed to \
            score. 
        * batch_normalization (bool): Whether to add batch normalization to the model. 

    Returns:
        * Model: The built model
    """
    
    # Define sequential models n_alpha and n_beta, as in the paper
    n_alpha = simple_deep_dense_net(training_data.shape[1:], 
                                    hidden_layers=hidden_layers, 
                                    dropout_frac=dropout_frac, 
                                    batch_normalization=batch_normalization)
    n_alpha._name = "n_alpha"
    n_beta = simple_deep_dense_net(training_data.shape[1:], 
                                   hidden_layers=hidden_layers, 
                                   dropout_frac=dropout_frac, 
                                   batch_normalization=batch_normalization)#,
                                   #final_activation="exponential")
    n_beta._name = "n_beta"

    # Define the functional model that makes use of n_alpha and n_beta to train them
    inputs = k.Input(shape=training_data.shape[1:])
    norm_layer = k.layers.Normalization()
    norm_layer.adapt(training_data)
    norm_layer = norm_layer(inputs)
    n_alpha_output = n_alpha(norm_layer)
    n_beta_output = n_beta(norm_layer)
    joined_output = k.layers.Concatenate()((n_alpha_output, n_beta_output))
    full_model = k.models.Model(inputs=inputs, outputs=[joined_output])
    
    # Compile the model
    full_model.compile(optimizer=optimizer, loss=build_score_loss(training_coefs, MSE_frac, score_subfrac))
    
    return full_model


def train_model(model: k.models.Model, 
                training_data: tf.Tensor, 
                true: tf.Tensor, 
                batch_size: int | None, 
                epochs: int, 
                patience: int, 
                min_delta: float, 
                reduce_lr_on_plateau: bool, 
                checkpoint_path: str | None, 
                checkpoint_frequency: int, 
                validation_split: float) -> k.callbacks.History:
    """Train the model and return its history

    Args:
        * model (k.models.Model): The model to train
        * training_data (tf.Tensor): The input data on which to train the model
        * true (tf.Tensor): The true values for calculating the loss
        * batch_size (int | None): The size of the batched. If None, it is set to the size of the entire dataset
        * epochs (int): The number of epochs for which to train
        * patience (int): The number of epochs for which loss has to stay constant before the training is stopped early
        * min_delta (float): The minimum change in loss to prevent early stopping
        * reduce_lr_on_plateau (bool): Whether to reduce the learning rate on plateaus. 
        * checkpoint_path (str | None): The path to the directory where you want to store checkpoints. \
            If None, no checkpoints are stored
        * checkpoint_frequency (int): The number of epochs to wait between saves.
        * validation_split (float): The fraction of the input training data to use for validation

    Returns:
        * k.callbacks.History: The history of the training
    """
    
    # Fit the model
    print(f"Reducing LR: {reduce_lr_on_plateau}")
    print(f"Checkpointing: {checkpoint_path is not None}")
    print(f"Checkpoint Frequency: {checkpoint_frequency}")
    return model.fit(x=training_data,
                     y=true,
                     shuffle=True,
                     batch_size=batch_size if batch_size is not None else training_data.shape[0],
                     epochs=epochs,
                     callbacks=[k.callbacks.EarlyStopping("val_loss" if validation_split > 0.0 else "loss", 
                                                          patience=patience, 
                                                          min_delta=min_delta, 
                                                          restore_best_weights=True), 
                                *((k.callbacks.ReduceLROnPlateau(monitor="val_loss" if validation_split > 0.0 else "loss", 
                                                                 patience=100, 
                                                                 min_delta=0.0, 
                                                                 min_lr=0.00001),)
                                  if reduce_lr_on_plateau else tuple()), 
                                *((EpochModelCheckpoint(filepath=checkpoint_path, 
                                                        monitor="val_loss" if validation_split > 0.0 else "loss", 
                                                        mode="min", 
                                                        save_best_only=True, 
                                                        save_weights_only=False, 
                                                        frequency=checkpoint_frequency),) 
                                  if checkpoint_path is not None else tuple())], 
                     validation_split=validation_split)


def save_run(model: k.models.Model, 
             history: k.callbacks.History, 
             network_name: str, 
             save_directory: str) -> None:
    """Save the model and the plots of the metrics.

    Args:
        * model (k.models.Model): The model to be saved.
        * history (k.callbacks.History): The history to use for plotting metrics.
        * network_name (str): The name with which to save the network.
        * save_directory (str): The path to the directory in which to save the network.
    """
    
    # Plot the loss over time and save
    plot_metrics(history=history, network_name=network_name, save_directory=save_directory)
    
    # Save the models
    print("MESSAGE TO USER: Do not worry if you get messages saying compiled metrics have yet to be built. "
          "They are not supposed to be built")
    save_models(model=model, network_name=network_name, save_directory=save_directory)


def plot_metrics(history: k.callbacks.History, network_name: str, save_directory: str) -> None:
    """Plot and save the loss function for a given network

    Args:
        * history (History): The history to use for plotting the loss
        * network_name (str): The name to give the plot
        * save_directory (str): The directory in which to save the plot
    """
    
    # Plot training and validation loss
    plt.figure(constrained_layout=True)
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history.keys():
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.yscale("log" if min(np.min(history.history["loss"]), np.min(history.history["val_loss"])) > 0.0 else "linear")
    else:
        plt.yscale("log" if np.min(history.history["loss"]) > 0.0 else "linear")
    plt.title(network_name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_directory}/{network_name}_loss.png")


def save_models(model: k.models.Model, network_name: str, save_directory: str) -> None:
    """Save the trained model and its submodels

    Args:
        * model (Model): The model to save
        * network_name (str): The name to give the network
        * save_directory (str): The directory in which to save the models
    """
    
    # Save the model
    model.save(f"{save_directory}/{network_name}")

    # Extract the alpha and beta subnetworks and save them
    n_alpha_trained = extract_submodel(model, ("input_1", "normalization", "n_alpha"))
    n_alpha_trained.save(f"{save_directory}/{network_name}_n_alpha")

    n_beta_trained = extract_submodel(model, ("input_1", "normalization", "n_beta"))
    n_beta_trained.save(f"{save_directory}/{network_name}_n_beta")


def extract_submodel(model: k.models.Model, layers: tuple[str, ...]) -> k.models.Model:
    """Return a sequential model built from the selected layers

    Args:
        * model (Model): The model to extract layers from
        * layers (tuple[str, ...]): The names of the layers to extract

    Returns:
        * Model: A sequential model built from the chosen layers
    """
    
    submodel = k.models.Sequential()
    for layer in layers:
        submodel.add(model.get_layer(layer))
    return submodel


def make_comparisons(n_alpha: k.models.Model, 
                     n_beta: k.models.Model,
                     coefficients: ty.Iterable, 
                     val_input: tf.Tensor, 
                     val_ratios: tf.Tensor, 
                     event_weights: tf.Tensor, 
                     network_name: str, 
                     save_directory: str) -> None:
    """Make log cross-section ratio plots to compare predictions to true results.

    Args:
        n_alpha (k.models.Model): The n_alpha model
        n_beta (k.models.Model): The n_beta model
        coefficients (ty.Iterable): The coefficient values on which to evaluate the model
        val_input (tf.Tensor): The validation input data
        val_ratios (tf.Tensor): The true validation ratios
        event_weights (tf.Tensor): The weights for each event
        network_name (str): The name with which to save the plots
        save_directory (str): The path to the directory in which to save the plots
    """
    
    # Get a list of the estimated cross-section ratios
    outputs = [(1.0 + coef * tf.squeeze(n_alpha(val_input))) ** 2.0 
               + (coef * tf.squeeze(n_beta(val_input))) ** 2.0 for coef in coefficients]
    
    # Get normalized event weights
    normed_event_weights = tf.squeeze(event_weights) / tf.math.reduce_sum(event_weights)
    
    # Same but with log ratios instead of ratios
    root_mean_square_log_error = [np.sqrt(np.sum(np.asarray(normed_event_weights, dtype=float) 
                                                                  * (np.log(np.asarray(output, dtype=float)) 
                                                                     - np.log(np.asarray(val_ratio, dtype=float))) ** 2.0)) 
                                  for output, val_ratio in zip(outputs, val_ratios)]
    
    # Plot the log cross-section ratios
    for output, val_ratio, log_error, coef in zip(outputs, val_ratios, root_mean_square_log_error, coefficients):
        plt.figure(constrained_layout=True)
        plt.axline((0.0, 0.0), slope=1.0, ls="--", c="r", zorder=0)
        plt.hist2d(np.log(np.squeeze(val_ratio)), np.log(np.squeeze(output)), bins=1000, weights=np.squeeze(event_weights), norm=mpl.colors.LogNorm())
        plt.xlabel("True log of Cross Section Ratios")
        plt.ylabel("Estimated log of Cross Section Ratios")
        plt.title(f"{network_name} at c = {coef}\nRMSLE = {log_error:.5}")
        plt.colorbar()
        plt.savefig(f"{save_directory}/{network_name}_ratios_{float_to_string(coef)}.png")


# Call the main function
if __name__ == "__main__":
    main()