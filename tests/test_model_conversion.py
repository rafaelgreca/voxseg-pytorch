import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tensorflow.keras import layers
from typing import Tuple

def _compare_convolutional1(signal: torch.tensor,
                            signal_numpy: np.ndarray) -> Tuple[torch.tensor, np.ndarray]:
    """
    Compares the Convolutional 1 layer output from the PyTorch and Keras model.

    Args:
        signal (torch.tensor): audio sample for the PyTorch model.
        signal_numpy (np.ndarray): audio sample for the Keras model.

    Returns:
        Tuple[torch.tensor, np.ndarray]: the PyTorch and Keras output.
    """
    
    # PyTorch model
    convolutional1 = TimeDistributed(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
        batch_first=True,
        layer_name="convolutional",
    )
    max_pooling = TimeDistributed(
        nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
    )
    
    x = F.relu(convolutional1(signal))
    output_torch = max_pooling(x)
    
    # Keras model
    convolutional1 = layers.TimeDistributed(
        layers.Conv2D(64, (5, 5), activation='elu'),
        input_shape=(None, 32, 32, 1)
    )
    max_pooling = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2))
    )

    x = convolutional1(signal_numpy)
    output_keras = max_pooling(x)
    
    return output_torch, output_keras

def _compare_convolutional2(signal: torch.tensor,
                            signal_numpy: np.ndarray) -> Tuple[torch.tensor, np.ndarray]:
    """
    Compares the Convolutional 2 layer output from the PyTorch and Keras model.

    Args:
        signal (torch.tensor): audio sample for the PyTorch model.
        signal_numpy (np.ndarray): audio sample for the Keras model.

    Returns:
        Tuple[torch.tensor, np.ndarray]: the PyTorch and Keras output.
    """
    
    # PyTorch model
    convolutional2 = TimeDistributed(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        batch_first=True,
        layer_name="convolutional",
    )
    max_pooling = TimeDistributed(
        nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
    )
    
    x = F.relu(convolutional2(signal))
    output_torch = max_pooling(x)
    
    # Keras model
    convolutional2 = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), activation='elu')
    )
    max_pooling = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2))
    )

    x = convolutional2(signal_numpy)
    output_keras = max_pooling(x)
    
    return output_torch, output_keras

def _compare_convolutional3(signal: torch.tensor,
                            signal_numpy: np.ndarray) -> Tuple[torch.tensor, np.ndarray]:
    """
    Compares the Convolutional 3 layer output from the PyTorch and Keras model.

    Args:
        signal (torch.tensor): audio sample for the PyTorch model.
        signal_numpy (np.ndarray): audio sample for the Keras model.

    Returns:
        Tuple[torch.tensor, np.ndarray]: the PyTorch and Keras output.
    """
    
    # PyTorch model
    convolutional3 = TimeDistributed(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
        batch_first=True,
        layer_name="convolutional",
    )
    max_pooling = TimeDistributed(
        nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
    )
    
    x = F.relu(convolutional3(signal))
    output_torch = max_pooling(x)
    
    # Keras model
    convolutional3 = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), activation='elu')
    )
    max_pooling = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2))
    )

    x = convolutional3(signal_numpy)
    output_keras = max_pooling(x)
    
    return output_torch, output_keras

def _compare_flatten(signal: torch.tensor,
                     signal_numpy: np.ndarray) -> Tuple[torch.tensor, np.ndarray]:
    """
    Compares the Flatten layer output from the PyTorch and Keras model.

    Args:
        signal (torch.tensor): audio sample for the PyTorch model.
        signal_numpy (np.ndarray): audio sample for the Keras model.

    Returns:
        Tuple[torch.tensor, np.ndarray]: the PyTorch and Keras output.
    """
    
    # PyTorch model
    flatten = TimeDistributed(
        nn.Flatten(), batch_first=True, layer_name="flatten"
    )

    output_torch = flatten(signal)
                
    # Keras model
    flatten = layers.TimeDistributed(
        layers.Flatten()
    )
    
    output_keras = flatten(signal_numpy)
    
    return output_torch, output_keras

def _compare_dense1(signal: torch.tensor,
                    signal_numpy: np.ndarray) -> Tuple[torch.tensor, np.ndarray]:
    """
    Compares the Dense layer output from the PyTorch and Keras model.

    Args:
        signal (torch.tensor): audio sample for the PyTorch model.
        signal_numpy (np.ndarray): audio sample for the Keras model.

    Returns:
        Tuple[torch.tensor, np.ndarray]: the PyTorch and Keras output.
    """
    
    # PyTorch model
    dense1 = TimeDistributed(
        nn.Linear(in_features=512, out_features=128),
        batch_first=True,
        layer_name="dense",
    )
    dropout = nn.Dropout(p=0.5)
    
    x = F.relu(dense1(signal))
    output_torch = dropout(x)
        
    # Keras model
    dense1 = layers.TimeDistributed(
        layers.Dense(128, activation='elu')
    )
    dropout = layers.Dropout(0.5)
    
    x = dense1(signal_numpy)
    output_keras = dropout(x)
    
    return output_torch, output_keras

def _compare_dense2(signal: torch.tensor,
                    signal_numpy: np.ndarray) -> Tuple[torch.tensor, np.ndarray]:
    """
    Compares the Final layer output from the PyTorch and Keras model.

    Args:
        signal (torch.tensor): audio sample for the PyTorch model.
        signal_numpy (np.ndarray): audio sample for the Keras model.

    Returns:
        Tuple[torch.tensor, np.ndarray]: the PyTorch and Keras output.
    """
    
    # PyTorch model
    dense2 = TimeDistributed(
        nn.Linear(in_features=256, out_features=2),
        batch_first=True,
        layer_name="dense",
    )
    dropout = nn.Dropout(p=0.5)
    
    x = dropout(signal)
    output_torch = F.relu(dense2(x))
        
    # Keras model
    dense2 = layers.TimeDistributed(
        layers.Dense(2, activation='softmax')
    )
    dropout = layers.Dropout(0.5)
    
    x = dropout(signal_numpy)
    output_keras = dense2(x)
    
    return output_torch, output_keras

def _compare_lstm(signal: torch.tensor,
                  signal_numpy: np.ndarray) -> Tuple[torch.tensor, np.ndarray]:
    """
    Compares the BiLSTM layer output from the PyTorch and Keras model.

    Args:
        signal (torch.tensor): audio sample for the PyTorch model.
        signal_numpy (np.ndarray): audio sample for the Keras model.

    Returns:
        Tuple[torch.tensor, np.ndarray]: the PyTorch and Keras output.
    """
    
    # PyTorch model
    bilstm = nn.LSTM(
        input_size=128,
        hidden_size=128,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
    )
    output_torch, _ = bilstm(signal)
    
    # Keras model
    bilstm = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True)
    )
    output_keras = bilstm(signal_numpy)
    
    return output_torch, output_keras

# All credits to: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first, layer_name):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
        self.layer_name = layer_name

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-3), x.size(-2), x.size(-1)
        )

        y = self.module(x_reshape)

        if self.layer_name == "convolutional" or self.layer_name == "max_pooling":

            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(
                    x.size(0), x.size(1), y.size(-3), y.size(-2), y.size(-1)
                )
            else:
                y = y.view(
                    -1, x.size(1), y.size(-1)
                )

        else:

            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(
                    x.size(0), x.size(1), y.size(-1)
                )
            else:
                y = y.view(
                    -1, x.size(1), y.size(-1)
                )

        return y
    
def voxseg_pytorch(signal: torch.tensor) -> torch.tensor:
    """
    Creates the Voxseg model using PyTorch.

    Args:
        signal (torch.tensor): the audio sample.

    Returns:
        x (torch.tensor): the model output.
    """
    # Defining the layers
    convolutional1 = TimeDistributed(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
        batch_first=True,
        layer_name="convolutional",
    )
    convolutional2 = TimeDistributed(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        batch_first=True,
        layer_name="convolutional",
    )
    convolutional3 = TimeDistributed(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
        batch_first=True,
        layer_name="convolutional",
    )
    max_pooling = TimeDistributed(
        nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
    )
    dense1 = TimeDistributed(
        nn.Linear(in_features=512, out_features=128),
        batch_first=True,
        layer_name="dense",
    )
    dense2 = TimeDistributed(
        nn.Linear(in_features=256, out_features=2),
        batch_first=True,
        layer_name="dense",
    )
    dropout = nn.Dropout(p=0.5)
    bilstm = nn.LSTM(
        input_size=128,
        hidden_size=128,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
    )
    flatten = TimeDistributed(
        nn.Flatten(), batch_first=True, layer_name="flatten"
    )

    ## first convolutional block
    x = F.relu(convolutional1(signal))
    x = max_pooling(x)

    ## second convolutional block
    x = F.relu(convolutional2(x))
    x = max_pooling(x)

    ## third convolutional block
    x = F.relu(convolutional3(x))
    x = max_pooling(x)

    ## flatten
    x = flatten(x)

    ## first dense layer
    x = F.relu(dense1(x))

    ## dropout
    x = dropout(x)

    ## bilstm
    x, _ = bilstm(x)

    ## dropout
    x = dropout(x)

    ## final layer
    x = dense2(x)
    
    return x
        
def voxseg_keras(signal: np.ndarray) -> np.ndarray:
    """
    Creates the Voxseg model using Tensorflow/Keras (the original code).

    Args:
        signal (np.darray): the audio sample.

    Returns:
        x (np.darray): model output.
    """
    
    # Defining the layers
    convolutional1 = layers.TimeDistributed(
        layers.Conv2D(64, (5, 5), activation='relu'),
        input_shape=(None, 32, 32, 1)
    )
    convolutional2 = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), activation='relu')
    )
    convolutional3 = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), activation='relu')
    )
    max_pooling = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2))
    )
    max_pooling2 = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2))
    )
    max_pooling3 = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2))
    )
    flatten = layers.TimeDistributed(
        layers.Flatten()
    )
    dense1 = layers.TimeDistributed(
        layers.Dense(128, activation='relu')
    )
    dropout = layers.Dropout(0.5)
    bilstm = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True)
    )
    dense2 = layers.TimeDistributed(
        layers.Dense(2, activation='softmax')
    )
    
    x = convolutional1(signal)
    
    x = max_pooling(x)
    
    x = convolutional2(x)
    
    x = max_pooling2(x)
    
    x = convolutional3(x)
    
    x = max_pooling3(x)
    
    x = flatten(x)
    
    x = dense1(x)
    
    x = dropout(x)
    
    x = bilstm(x)
    
    x = dropout(x)
    
    x = dense2(x)
    
    return x
    
@pytest.mark.parametrize("signal", [
    (torch.rand(size=(50, 15, 1, 32, 32), dtype=torch.float32)),
    (torch.rand(size=(150, 15, 1, 32, 32), dtype=torch.float32)),
    (torch.rand(size=(10, 15, 1, 32, 32), dtype=torch.float32))
])
def test1(signal):
    """
    Test if the models output (PyTorch and Keras) has the same shape.

    Args:
        signal (torch.tensor): the audio sample.
    """
    signal_numpy = signal.permute(0, 1, 3, 4, 2).detach().numpy()
    output_model_keras = voxseg_keras(signal_numpy)
    output_model_pytorch = voxseg_pytorch(signal)
    
    assert output_model_pytorch.shape == output_model_keras.shape

@pytest.mark.parametrize("signal", [
    (torch.rand(size=(30, 15, 1, 32, 32), dtype=torch.float32)),
    (torch.rand(size=(90, 15, 1, 32, 32), dtype=torch.float32)),
    (torch.rand(size=(54, 15, 1, 32, 32), dtype=torch.float32))
])
def test2(signal):
    """
    Test if the models output (PyTorch and Keras) has the same shape.

    Args:
        signal (torch.tensor): the audio sample.
    """
    signal_numpy = signal.permute(0, 1, 3, 4, 2).detach().numpy()
    output_torch, output_keras = _compare_convolutional1(signal=signal,
                                                         signal_numpy=signal_numpy)
    
    assert output_torch.detach().permute(0, 1, 3, 4, 2).numpy().shape == output_keras.shape
    
    output_torch, output_keras = _compare_convolutional2(signal=output_torch,
                                                         signal_numpy=output_keras)
    
    assert output_torch.detach().permute(0, 1, 3, 4, 2).numpy().shape == output_keras.shape
    
    output_torch, output_keras = _compare_convolutional3(signal=output_torch,
                                                         signal_numpy=output_keras)
    
    assert output_torch.detach().permute(0, 1, 3, 4, 2).numpy().shape == output_keras.shape
    
    output_torch, output_keras = _compare_flatten(signal=output_torch,
                                                  signal_numpy=output_keras)
    
    assert output_torch.shape == output_keras.shape
    
    output_torch, output_keras = _compare_dense1(signal=output_torch,
                                                 signal_numpy=output_keras)
    
    assert output_torch.shape == output_keras.shape
    
    output_torch, output_keras = _compare_lstm(signal=output_torch,
                                               signal_numpy=output_keras)
    
    assert output_torch.shape == output_keras.shape
    
    output_torch, output_keras = _compare_dense2(signal=output_torch,
                                                 signal_numpy=output_keras)
    
    assert output_torch.shape == output_keras.shape