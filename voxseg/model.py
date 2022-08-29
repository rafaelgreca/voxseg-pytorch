import os
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        )  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        if self.layer_name == "convolutional" or self.layer_name == "max_pooling":

            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(
                    x.size(0), x.size(1), y.size(-3), y.size(-2), y.size(-1)
                )  # (samples, timesteps, output_size)
            else:
                y = y.view(
                    -1, x.size(1), y.size(-1)
                )  # (timesteps, samples, output_size)

        else:

            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(
                    x.size(0), x.size(1), y.size(-1)
                )  # (samples, timesteps, output_size)
            else:
                y = y.view(
                    -1, x.size(1), y.size(-1)
                )  # (timesteps, samples, output_size)

        return y


class Voxseg(nn.Module):
    def __init__(self, num_labels):
        super(Voxseg, self).__init__()
        self.convolutional1 = TimeDistributed(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
            batch_first=True,
            layer_name="convolutional",
        )
        self.convolutional2 = TimeDistributed(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            batch_first=True,
            layer_name="convolutional",
        )
        self.convolutional3 = TimeDistributed(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            batch_first=True,
            layer_name="convolutional",
        )
        self.max_pooling = TimeDistributed(
            nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
        )
        self.dense1 = TimeDistributed(
            nn.Linear(in_features=512, out_features=128),
            batch_first=True,
            layer_name="dense",
        )
        self.dense2 = TimeDistributed(
            nn.Linear(in_features=256, out_features=num_labels),
            batch_first=True,
            layer_name="dense",
        )
        self.dropout = nn.Dropout(p=0.5)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.flatten = TimeDistributed(
            nn.Flatten(), batch_first=True, layer_name="flatten"
        )

    def forward(self, x):
        ## first convolutional block
        x = F.relu(self.convolutional1(x))
        x = self.max_pooling(x)

        ## second convolutional block
        x = F.relu(self.convolutional2(x))
        x = self.max_pooling(x)

        ## third convolutional block
        x = F.relu(self.convolutional3(x))
        x = self.max_pooling(x)

        ## flatten
        x = self.flatten(x)

        ## first dense layer
        x = F.relu(self.dense1(x))

        ## dropout
        x = self.dropout(x)

        ## bilstm
        x, _ = self.bilstm(x)

        ## dropout
        x = self.dropout(x)

        ## final layer
        x = self.dense2(x)

        return x


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, output_dir: str, model_name: str):
        self.best_valid_loss = float("inf")
        self.output_dir = output_dir
        self.model_name = model_name

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch}\n")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                os.path.join(self.output_dir, f"{self.model_name}.pth"),
            )
