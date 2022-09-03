import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import tensorflow as tf
import numpy as np
import random
import os
from voxseg import utils, prep_labels, extract_feats
from voxseg.model import TimeDistributed, Voxseg, weight_init
from voxseg.dataset import AVA_Dataset
from torch.utils.data import DataLoader
from tensorflow.keras import models, layers, losses
from keras.utils.layer_utils import count_params

torch.set_num_threads(1)
torch.manual_seed(21)
torch.backends.cudnn.benchmark = False
random.seed(21)
np.random.seed(21)
tf.random.set_seed(21)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def seed_worker(worker_id):
    np.random.seed(21)
    random.seed(21)

g = torch.Generator()
g.manual_seed(21)

# All credits to: https://stackoverflow.com/questions/65383500/tensorflow-keras-keep-loss-of-every-batch
batches_loss_keras = list()
class SaveBatchLoss(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        batches_loss_keras.append(logs['loss'])
        
# Original model
def cnn_bilstm(output_layer_width):
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Conv2D(64, (5, 5), activation='elu'), input_shape=(None, 32, 32, 1)))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='elu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='elu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dense(128, activation='elu')))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(output_layer_width, activation='softmax')))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def test1():
    # Creating the models
    voxseg_pytorch = Voxseg(2)
    voxseg_pytorch.apply(weight_init)
    voxseg_keras = cnn_bilstm(2)

    path = os.path.join(os.getcwd(), "tests")
    
    # Preprocessing the data
    data = prep_labels.prep_data(path)
    feats = extract_feats.extract(data)
    feats = extract_feats.normalize(feats)
    labels = prep_labels.get_labels(data)
    labels["labels"] = prep_labels.one_hot(labels["labels"])

    # Train model
    X = utils.time_distribute(np.vstack(feats["normalized-features"]), 15)
    y = utils.time_distribute(np.vstack(labels["labels"]), 15)

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_torch = torch.from_numpy(X)
    X_torch = X_torch.unsqueeze(2)
    y_torch = torch.from_numpy(y)
        
    # PyTorch model
    output_pytorch = voxseg_pytorch(X_torch)
    loss_pytorch = F.binary_cross_entropy(output_pytorch, y_torch)
    
    # Keras model
    output_keras = voxseg_keras(X[:, :, :, :, np.newaxis])
    loss_fn = losses.BinaryCrossentropy()
    loss_keras = loss_fn(y, output_keras)
    
    assert loss_keras.shape == loss_pytorch.shape

def test2():
    """_summary_
    """
    # Creating the models
    voxseg_pytorch = Voxseg(2)
    voxseg_pytorch.apply(weight_init)
    voxseg_keras = cnn_bilstm(2)
    
    total_params_torch = sum(
        param.numel() for param in voxseg_pytorch.parameters()
    )

    trainable_params_torch = sum(
        p.numel() for p in voxseg_pytorch.parameters() if p.requires_grad
    )

    trainable_params_keras = sum(count_params(layer) for layer in voxseg_keras.trainable_weights)
    non_trainable_params_keras = sum(count_params(layer) for layer in voxseg_keras.non_trainable_weights)
    total_params_keras = trainable_params_keras + non_trainable_params_keras

    assert (total_params_keras * 0.95) <= total_params_torch <= (total_params_keras * 1.05)
    assert (trainable_params_keras * 0.95) <= trainable_params_torch <= ((trainable_params_keras * 1.05))
    
# TODO: Compare the training step of both models  
def test3():
    """_summary_
    """
    # Creating the models
    voxseg_pytorch = Voxseg(2)
    voxseg_pytorch.apply(weight_init)
    voxseg_keras = cnn_bilstm(2)
    
    # Initializing the variables
    epochs = 1
    batch_size = 8
    training_log = pd.DataFrame()
    use_shuffle = False
    path = os.path.join(os.getcwd(), "tests")
    optimizer = optim.Adam(voxseg_pytorch.parameters())
    
    # Preprocessing the data
    data = prep_labels.prep_data(path)
    feats = extract_feats.extract(data)
    feats = extract_feats.normalize(feats)
    labels = prep_labels.get_labels(data)
    labels["labels"] = prep_labels.one_hot(labels["labels"])

    # Train model
    X = utils.time_distribute(np.vstack(feats["normalized-features"]), 15)
    y = utils.time_distribute(np.vstack(labels["labels"]), 15)

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Training the PyTorch model
    dataset = AVA_Dataset(X=X,
                          y=y)
    
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=use_shuffle,
                            num_workers=0,
                            worker_init_fn=seed_worker,
                            generator=g)
    
    voxseg_pytorch.train()
    
    for epoch in range(1, epochs+1):
        
        for idx, batch in enumerate(dataloader):
            data = batch["X"]
            data = data.unsqueeze(2)
            target = batch["y"]

            optimizer.zero_grad()
            output = voxseg_pytorch(data)

            l = F.binary_cross_entropy(output, target)
            l.backward()
            optimizer.step()
            
            temp = pd.DataFrame({
                "epoch": [epoch],
                "batch": [idx],
                "loss_pytorch": [l.item()]
            })
            
            training_log = pd.concat([training_log, temp], axis=0)
            
            del temp
    
    training_log = training_log.reset_index(drop=True)
    
    # Training the Keras model
    voxseg_keras =  voxseg_keras.fit(X[:, :, :, :, np.newaxis],
                                     y,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     shuffle=use_shuffle,
                                     callbacks=SaveBatchLoss(),
                                     max_queue_size=1)

    training_log["loss_keras"] = batches_loss_keras
        
    assert training_log.iloc[0, 2] > training_log.iloc[-1, 2]
    assert training_log.iloc[0, 3] > training_log.iloc[-1, 3]