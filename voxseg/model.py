import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# All credits to: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346
class TimeDistributed(nn.Module):
    """
    Mimics the Keras TimeDistributed layer.
    """

    def __init__(self, module, batch_first, layer_name):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
        self.layer_name = layer_name

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3), x.size(-2), x.size(-1))

        y = self.module(x_reshape)

        if self.layer_name == "convolutional" or self.layer_name == "max_pooling":

            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(
                    x.size(0), x.size(1), y.size(-3), y.size(-2), y.size(-1)
                )
            else:
                y = y.view(-1, x.size(1), y.size(-1))

        else:

            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(x.size(0), x.size(1), y.size(-1))
            else:
                y = y.view(-1, x.size(1), y.size(-1))

        return y


# All credits to: https://discuss.pytorch.org/t/crossentropyloss-expected-object-of-type-torch-longtensor/28683/6?u=ptrblck
def weight_init(m):
    """
    Initalize all the weights in the PyTorch model to be the same as Keras.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_ih_l0, gain=nn.init.calculate_gain("relu"))
        nn.init.orthogonal_(m.weight_hh_l0)
        nn.init.zeros_(m.bias_ih_l0)
        nn.init.zeros_(m.bias_hh_l0)


class Extract_LSTM_Output(nn.Module):
    """
    Extracts only the output from the BiLSTM layer.
    """

    def forward(self, x):
        output, _ = x
        return output


class Voxseg(nn.Module):
    """
    Creates the Voxseg model in PyTorch.
    """

    def __init__(self, num_labels):
        super(Voxseg, self).__init__()
        self.layers = nn.Sequential(
            TimeDistributed(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
                batch_first=True,
                layer_name="convolutional",
            ),
            nn.ELU(),
            TimeDistributed(
                nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
            ),
            TimeDistributed(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                batch_first=True,
                layer_name="convolutional",
            ),
            nn.ELU(),
            TimeDistributed(
                nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
            ),
            TimeDistributed(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
                batch_first=True,
                layer_name="convolutional",
            ),
            nn.ELU(),
            TimeDistributed(
                nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
            ),
            TimeDistributed(nn.Flatten(), batch_first=True, layer_name="flatten"),
            TimeDistributed(
                nn.Linear(in_features=512, out_features=128),
                batch_first=True,
                layer_name="dense",
            ),
            nn.Dropout(p=0.5),
            nn.LSTM(
                input_size=128,
                hidden_size=128,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            ),
            Extract_LSTM_Output(),
            nn.Dropout(p=0.5),
            TimeDistributed(
                nn.Linear(in_features=256, out_features=num_labels),
                batch_first=True,
                layer_name="dense",
            ),
        )

    def forward(self, x):
        return self.layers(x)


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, output_dir: str, model_name: str):
        self.best_valid_loss = float("inf")
        self.best_valid_acc = 0.0
        self.output_dir = output_dir
        self.model_name = model_name

    def __call__(self, current_valid_loss, current_valid_acc, epoch, model, optimizer):
        if current_valid_acc > self.best_valid_acc:
            self.best_valid_loss = current_valid_loss
            self.best_valid_acc = current_valid_acc
            print("\nSaving best model...")
            print(f"Epoch: {epoch}")
            print(f"Validation accuracy: {self.best_valid_acc:1.6f}")
            print(f"Validation loss: {self.best_valid_loss:1.6f}\n")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(self.output_dir, f"{self.model_name}.pth"),
            )
