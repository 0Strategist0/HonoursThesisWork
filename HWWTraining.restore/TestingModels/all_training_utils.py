"""
# All Training Utils

---------------------------------------------------------------------------------------------------------------------------------

Kye Emond

June 15, 2023              
                                                                                                   
---------------------------------------------------------------------------------------------------------------------------------

A module designed to help training a neural network for cross-section ratio estimation, using all events to their maximum \
potential.

---------------------------------------------------------------------------------------------------------------------------------

### Classes:
* EpochModelCheckpoint: A subclasses of the checkpoint callback in Keras. Lets you set checkpoints after a certain \
    number of epochs instead of after a certain number of batches. 

### Functions:
* events_to_training: Takes in a pandas Dataframe of nTuples and returns two sets of data. 
    The first array of data is the set of kinematic variables to be used as input for the neural network. 
    The second array of data is extra data used in the loss functions during training. In most cases it's the SM weights \
    followed by the EFT weights, but if use_alpha_beta_loss is True, it's the INT/SM, BSM/SM, SM weights. 
* float_to_string: Converts a float to a string in the format used for the nTuples.
* string_to_float: Converts a string in the format of the nTuples into a float.
* simple_deep_dense_net: Quickly build a deep, densely connected neural network with the desired parameters.
* f: Used for some loss functions.
* f_direct: Used for some other loss functions.
* build_loss: Builds the quadratic loss function recommended in the paper depending on the coefficient values used \
    for training.
* build_direct_square_loss: Builds the mean squared error loss function of the direct output, rather than the reciprocal.
* build_maximum_likelihood_loss: Builds the maximum likelihood loss function.
* build_alpha_beta_loss: Builds a loss function that tries to minimize the mean squared error of the alpha and beta networks, \
    rather than of the differential cross-section ratio. 
"""

# Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as k
import typing as ty

# Useful functions for preprocessing
def events_to_training(events: pd.DataFrame, 
                       coef_values: ty.Iterable[float],
                       c_value_prefix: str = "weight_cHW_", 
                       SM_weight_title: str = "weight_sm", 
                       kinematic_columns: np.ndarray = np.arange(2, 39), 
                       randomize_data: bool = True, 
                       use_alpha_beta_loss: bool = False, 
                       int_prefix: str | None = None, 
                       bsm_prefix: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the training data as well as an array of coefficients and weights to use for calculating loss

    Args:
        * events (pd.DataFrame): The events used for training
        * coef_values (Iterable[float]): The coefficient values you want to use for fitting
        * c_value_prefix (str, optional): The prefix to the titles of the weight columns. Defaults to "weight_cHj3_".
        * SM_weight_title (str, optional): The title of the standard model weight column. Defaults to "weight_sm".
        * kinematic_columns (np.ndarray, optional): An array of indices that have the kinematic variables. \
            Defaults to np.arange(2, 22).
        * randomize_data (bool, optional): Whether to randomize the output order of the data. Defaults to True.
        * use_alpha_beta_loss (bool, optional): Whether to use alpha-beta loss. Defaults to False. 
        * int_prefix (str | None): The prefix for interference weights. Only used if use_alpha_beta_loss is True. Defaults to None. 
        * bsm_prefix (str | None): The prefix for beyond standard model weight. Only used if use_alpha_beta_loss is True. \
            Defaults to None. 

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple holding the training data as well as the array used for loss calculation.
    """
    
    # Get a numpy array with all the kinematic variables per event
    return_kinematics = events.to_numpy()[:, kinematic_columns]
    
    # Get an array of the standard model weights for each event
    sm_weights = np.expand_dims(events[SM_weight_title].to_numpy(), 1)
    
    if not use_alpha_beta_loss:
        # Get an array of the SMEFT weights for each event
        all_weights_arr = np.array([events[c_value_prefix + float_to_string(value)] for value in coef_values]).T
        
        # Get an array of all the weights
        return_weights = np.concatenate((sm_weights, all_weights_arr), axis=1)
    else:
        # Get an array of the interference and beyond standard model weights for each event
        int_weights = np.array([events[int_prefix + float_to_string(value)] for value in coef_values]).T
        bsm_weights = np.array([events[bsm_prefix + float_to_string(value)] for value in coef_values]).T
        
        # Stack the two and divide by SM, sticking SM in the back there
        return_weights = np.stack((int_weights / sm_weights, 
                                   bsm_weights / sm_weights, 
                                   np.repeat(sm_weights, len(coef_values), axis=-1)), axis=-1)
    
    # Randomize the data
    shuffled_indices = np.random.permutation(len(return_kinematics)) if randomize_data else np.arange(len(return_kinematics))
    
    # Return the values calculated
    return return_kinematics[shuffled_indices], return_weights[shuffled_indices]
    

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


def string_to_float(string: str) -> float:
    """Return the float corresponding to the weird nTuple-formatted string.

    Args:
        string (str): The formatted string to convert.

    Returns:
        float: The float version of the string
    """
    
    number_bit = string.split("_")[-1].replace("pos", "").replace("neg", "-").replace("p", ".")
    return float(number_bit)
    

# Model building
def simple_deep_dense_net(input_shape: tuple, 
                          output_neurons: int = 1, 
                          hidden_layers: ty.Iterable[int] = (32, 32, 32, 32), 
                          activation: str = "relu", 
                          final_activation: str = "linear", 
                          dropout_frac: float = 0.1, 
                          batch_normalization: bool = True) -> k.models.Model:
    """Return a deep, densely connected neural network with the same topology per layer

    Args:
        * input_shape (tuple): The shape of the input data
        * output_neurons (int, optional): The number of output neurons. Defaults to 1.
        * hidden_layers (ty.Iterable, optional): An iterable of the neurons for each hidden layer. Defaults to (32, 32)
        * activation (str, optional): The activation type of the neurons. Defaults to "relu".
        * final_activation (str, optional): The final activation function. Defaults to "linear".
        * dropout_frac (float, optional): The fraction of neurons to be dropped every cycle. Defaults to 0.1.
        * batch_normalization (bool, optional): Whether to include batch normalization or not. Defaults to True.

    Returns:
        Model: A simple deep dense neural network
    """
    # Initialize the network
    net = k.models.Sequential()
    # Set up the input layer of the network
    net.add(k.layers.Dense(hidden_layers[0], activation, input_shape=input_shape))
    if batch_normalization:
        net.add(k.layers.BatchNormalization(synchronized=True))
    if dropout_frac > 0.0:
        net.add(k.layers.Dropout(dropout_frac))
    
    # Iterate through the layers provided by the user and add the correctly sized layer for each item
    for neurons in hidden_layers[1:]:
        net.add(k.layers.Dense(neurons, activation))
        if batch_normalization:
            net.add(k.layers.BatchNormalization(synchronized=True))
        if dropout_frac > 0.0:
            net.add(k.layers.Dropout(dropout_frac))
    net.add(k.layers.Dense(output_neurons, final_activation))
    
    return net

def f(n_alpha: tf.Tensor, n_beta: tf.Tensor, coefs: tf.Tensor) -> tf.Tensor:
    return 1.0 / (1.0 + (1.0 + coefs * n_alpha) ** 2.0 + (coefs * n_beta) ** 2.0)


def f_direct(n_alpha: tf.Tensor, n_beta: tf.Tensor, coefs: tf.Tensor) -> tf.Tensor:
    return (1.0 + coefs * n_alpha) ** 2.0 + (coefs * n_beta) ** 2.0


def build_score_loss(training_coefs: ty.Iterable, MSE_frac: float, score_subfrac: float) -> ty.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Return the direct square loss with the given weights and network outputs. 
    The direct square loss is given by weight * ([estimated cross-section ratio] - [actual cross-section ratio]) ** 2

    Args:
        training_coefs (ty.Iterable): An iterable containing the values of the Wilson Coefficient used for training.

    Returns:
        ty.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: The loss function to use for training
    """
    
    coefs_array = tf.reshape(training_coefs, (1, len(training_coefs)))
    
    def loss(weights_score_quad: tf.Tensor, network_output: tf.Tensor) -> tf.Tensor:
        """Return the loss evaluated with the given weights and network outputs

        Args:
            weights (tf.Tensor): The event weights, followed by score and quad. Axis 0 should be which event you're looking at. \
                Along axis 1, the 0th element should be the SM weight. Beyond that, the weights should \
                be in the same order as the coefs_array. 
            network_output (tf.Tensor): The output of the neural network. Axis 0 should be which event you're looking at. \
                In the 0th position of axis 1, you should have the alpha network output, and in the 1st position, the beta.

        Returns:
            tf.Tensor: The loss evaluated for each event.
        """
        
        f_value = tf.math.log(f_direct(tf.squeeze(network_output[:, 0])[:, tf.newaxis], 
                                  tf.squeeze(network_output[:, 1])[:, tf.newaxis], 
                                  coefs_array))
        
        mse = tf.math.reduce_sum((tf.abs(weights_score_quad[:, 0, tf.newaxis]) + tf.abs(weights_score_quad[:, 1:-2])) 
                                  * (f_value - tf.math.log(tf.abs(weights_score_quad[:, 1:-2] / weights_score_quad[:, 0, tf.newaxis]))) ** 2.0, 
                                  axis=1)
        
        score_err = tf.abs(weights_score_quad[:, 0]) * (2.0 * tf.squeeze(network_output[:, 0]) - weights_score_quad[:, -2]) ** 2.0
        
        quad_err = tf.abs(weights_score_quad[:, 0]) * (tf.squeeze(network_output[:, 0] ** 2.0 + network_output[:, 1] ** 2.0) - weights_score_quad[:, -1]) ** 2.0
        
        return MSE_frac * mse + (1.0 - MSE_frac) * (score_subfrac * score_err + (1.0 - score_subfrac) * quad_err)
        
    return loss


def build_loss(training_coefs: ty.Iterable) -> ty.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Build and return a loss function using the given training coefficients

    Args:
        training_coefs (ty.Iterable): An iterable containing the coefficients used for training

    Returns:
        ty.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: The loss function to use for training
    """
    
    # Get an appropriately reshaped array of coefficients
    coefs_array = tf.reshape(training_coefs, (1, len(training_coefs)))
    
    # Define the function to be returned
    def loss(weights: tf.Tensor, network_output: tf.Tensor) -> tf.Tensor:
        """Return the loss evaluated with the given weights and network outputs

        Args:
            weights (tf.Tensor): The event weights. Axis 0 should be which event you're looking at. \
                Along axis 1, the 0th element should be the SM weight. Beyond that, the weights should \
                be in the same order as the coefs_array. 
            network_output (tf.Tensor): The output of the neural network. Axis 0 should be which event you're looking at. \
                In the 0th position of axis 1, you should have the alpha network output, and in the 1st position, the beta.

        Returns:
            tf.Tensor: The loss evaluated for each event.
        """
        
        return tf.math.reduce_sum(
            (tf.abs(weights[:, 0, tf.newaxis]) + tf.abs(weights[:, 1:])) * 
            (f(tf.squeeze(network_output[:, 0])[:, tf.newaxis], tf.squeeze(network_output[:, 1])[:, tf.newaxis], coefs_array)
                 - 1.0 / (1.0 + tf.abs(weights[:, 1:] / weights[:, 0, tf.newaxis]))) ** 2.0, 
             axis=1)
    
    # Return the created loss function
    return loss


def build_direct_square_loss(training_coefs: ty.Iterable) -> ty.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Return the direct square loss with the given weights and network outputs. 
    The direct square loss is given by weight * ([estimated cross-section ratio] - [actual cross-section ratio]) ** 2

    Args:
        training_coefs (ty.Iterable): An iterable containing the values of the Wilson Coefficient used for training.

    Returns:
        ty.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: The loss function to use for training
    """
    
    coefs_array = tf.reshape(training_coefs, (1, len(training_coefs)))
    
    def loss(weights: tf.Tensor, network_output: tf.Tensor) -> tf.Tensor:
        """Return the loss evaluated with the given weights and network outputs

        Args:
            weights (tf.Tensor): The event weights. Axis 0 should be which event you're looking at. \
                Along axis 1, the 0th element should be the SM weight. Beyond that, the weights should \
                be in the same order as the coefs_array. 
            network_output (tf.Tensor): The output of the neural network. Axis 0 should be which event you're looking at. \
                In the 0th position of axis 1, you should have the alpha network output, and in the 1st position, the beta.

        Returns:
            tf.Tensor: The loss evaluated for each event.
        """
        
        f_value = tf.math.log(f_direct(tf.squeeze(network_output[:, 0])[:, tf.newaxis], 
                                  tf.squeeze(network_output[:, 1])[:, tf.newaxis], 
                                  coefs_array))
        
        return tf.math.reduce_sum((tf.abs(weights[:, 0, tf.newaxis]) + tf.abs(weights[:, 1:])) 
                                  * (f_value - tf.math.log(tf.abs(weights[:, 1:] / weights[:, 0, tf.newaxis]))) ** 2.0, 
                                  axis=1)
        
    return loss


def build_maximum_likelihood_loss(training_coefs: ty.Iterable) -> ty.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Return the maximum likelihood loss with the given weights and network outputs. 
    The maximum likelihood loss is given by 
    -((sigma_0) * ln(estimated cross-section ratio) + (sigma_1) * (1 - esimtated cross-section ratio))

    Args:
        training_coefs (ty.Iterable): An iterable containing the values of the Wilson Coefficient used for training.

    Returns:
        ty.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: The loss function to use for training
    """
    
    coefs_array = tf.reshape(training_coefs, (1, len(training_coefs)))
    
    def loss(weights: tf.Tensor, network_output: tf.Tensor) -> tf.Tensor:
        """Return the loss evaluated with the given weights and network outputs

        Args:
            weights (tf.Tensor): The event weights. Axis 0 should be which event you're looking at. \
                Along axis 1, the 0th element should be the SM weight. Beyond that, the weights should \
                be in the same order as the coefs_array. 
            network_output (tf.Tensor): The output of the neural network. Axis 0 should be which event you're looking at. \
                In the 0th position of axis 1, you should have the alpha network output, and in the 1st position, the beta.

        Returns:
            tf.Tensor: The loss evaluated for each event.
        """
        
        
        f_value = f_direct(tf.squeeze(network_output[:, 0])[:, tf.newaxis], 
                           tf.squeeze(network_output[:, 1])[:, tf.newaxis], 
                           coefs_array)
        
        return -tf.math.reduce_sum(tf.abs(weights[:, 1:]) * tf.math.log(f_value) 
                                  + tf.abs(weights[:, 0, tf.newaxis]) * (1.0 - f_value), axis=1)
    
    return loss


def build_alpha_beta_loss(training_coefs: ty.Iterable) -> ty.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Return the alpha/beta loss with the given weights and network outputs. 
    Alpha/beta loss is given by weight * ((n_alpha - alpha) ** 2 + (n_beta - beta) ** 2).

    Args:
        training_coefs (ty.Iterable): An iterable containing the values of the Wilson Coefficient used for training.

    Returns:
        ty.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]: The loss function to use for training
    """
    
    coefs_array = tf.reshape(training_coefs, (1, len(training_coefs)))
    
    def loss(weight_ratios: tf.Tensor, network_output: tf.Tensor) -> tf.Tensor:
        """Return the alpha/beta loss evaluated with the given weights and network outputs. 

        Args:
            weight_ratios (tf.Tensor): The event weight ratios. Should be a tensor of shape \
                (number of events, number of coefficients, 3). The [:, :, 0] elements should be interaction weights divided\
                by standard model weights, while the [:, :, 1] elements should be BSM divided by SM weights. \
                The [:, :, 2] elements should be the standard model weights. 
            network_output (tf.Tensor): The output of the neural network. Axis 0 should be which event you're looking at. \
                In the 0th position of axis 1, you should have the alpha network output, and in the 1st position, the beta.

        Returns:
            tf.Tensor: The loss evaluated for each event.
        """
        
        loss_weighting = tf.math.abs(2.0 * weight_ratios[:, :, 2] + tf.math.reduce_sum(weight_ratios[:, :, :2] * weight_ratios[:, :, 2, tf.newaxis], axis=2))
        
        alpha_chi_sq = tf.math.reduce_sum(
            loss_weighting * (tf.squeeze(network_output[:, 0])[:, tf.newaxis] - weight_ratios[:, :, 0] / (2.0 * coefs_array)) ** 2.0, 
            axis=1)
        
        beta_chi_sq = tf.math.reduce_sum(
            loss_weighting * (tf.squeeze(network_output[:, 1])[:, tf.newaxis] 
                                         - tf.math.sqrt(tf.math.maximum(weight_ratios[:, :, 1] 
                                                                        - (weight_ratios[:, :, 0] / 2.0) ** 2.0, 0.0)) 
                                         / tf.math.abs(coefs_array)) ** 2.0, 
            axis=1)
        
        return alpha_chi_sq + beta_chi_sq
    
    return loss

class EpochModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """A class to checkpoint models after a certain number of epochs instead of batches"""

    def __init__(self,
                 filepath: str,
                 frequency: int = 1,
                 monitor: str = "val_loss",
                 verbose: int = 0,
                 save_best_only: bool = False,
                 save_weights_only: bool = False,
                 mode: str = "auto",
                 options=None,
                 **kwargs):
        super(EpochModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                   mode, "epoch", options)
        self.epochs_since_last_save = 0
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.epochs_since_last_save % self.frequency == 0:
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        pass

