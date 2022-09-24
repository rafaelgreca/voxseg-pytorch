import numpy as np
import argparse
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from voxseg.model import Voxseg, SaveBestModel, weight_init
from voxseg.dataset import AVA_Dataset, Preprocessing_Dataset, custom_collate
from voxseg import utils, extract_feats, prep_labels
from torch.utils.data import DataLoader
from typing import Tuple

# Making sure the experiments are reproducible
seed = 2109
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)


def train(
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim,
    epoch: int,
    model: torch.nn.Module,
    loss: torch.nn.CrossEntropyLoss,
) -> Tuple[float, float]:
    """
    Function responsible for the training step.

    Args:
        device (torch.device): the device (cpu or cuda).
        train_loader (torch.utils.data.DataLoader): the training dataloader.
        optimizer (torch.nn.optim): the optimizer that will be used.
        epoch (int): the current epoch.
        model (torch.nn.Module): the Voxseg module.

    Returns:
        Tuple[float, float]: current epoch training loss and accuracy.
    """
    model.train()
    training_loss = 0
    training_acc = 0
    pctg_to_print = list(range(0, 101, 25))

    for batch_idx, batch in enumerate(train_loader):
        data = batch["X"]
        data = data.unsqueeze(2)
        target = batch["y"]

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        output = output.view(output.size(0) * output.size(1), output.size(2))
        target = target.view(target.size(0) * target.size(1), target.size(2))
        
        l = loss(output, target)
        l.backward()
        optimizer.step()

        training_loss += l.item()
                
        pred = output.argmax(dim=1).view(-1, 1)
        target_class = target.argmax(dim=1).view(-1, 1)
        training_acc += pred.eq(target_class).sum().item() / len(target_class)

        pctg_batch = 100.0 * (batch_idx + 1) / len(train_loader)

        if round(pctg_batch, 0) in pctg_to_print:
            if round(pctg_batch, 0) == 25:
                print(
                    "\nTrain Epoch: {} ({:.0f}%)".format(
                        epoch,
                        100.0 * (batch_idx + 1) / len(train_loader),
                    )
                )
            else:
                print(
                    "Train Epoch: {} ({:.0f}%)".format(
                        epoch,
                        100.0 * (batch_idx + 1) / len(train_loader),
                    )
                )
            pctg_to_print.remove(round(pctg_batch, 0))

    training_loss /= len(train_loader)
    training_acc /= len(train_loader)
    return training_loss, training_acc


def validation(
    device: torch.device, validation_loader: DataLoader, model: torch.nn.Module
) -> Tuple[float, float]:
    """
    Function responsible for the training step.

    Args:
        device (torch.device): the device (cpu or cuda).
        validation_loader (torch.utils.data.DataLoader): the validation dataloader.
        model (torch.nn.Module): the Voxseg module.

    Returns:
        Tuple[float, float]: current epoch validation loss and accuracy.
    """
    model.eval()
    validation_loss = 0
    validation_acc = 0

    with torch.no_grad():
        for batch in validation_loader:
            data = batch["X"]
            data = data.unsqueeze(2)
            target = batch["y"]

            data, target = data.to(device), target.to(device)
            output = model(data)
            
            output = output.view(output.size(0) * output.size(1), output.size(2))
            target = target.view(target.size(0) * target.size(1), target.size(2))
        

            l = loss(output, target)
            validation_loss += l.item()

            pred = output.argmax(dim=1).view(-1, 1)
            target_class = target.argmax(dim=1).view(-1, 1)
            validation_acc += pred.eq(target_class).sum().item() / len(
                target_class
            )

    validation_loss /= len(validation_loader)
    validation_acc /= len(validation_loader)
    return validation_loss, validation_acc

# Function responsible to preprocess the train and validation data
def preprocessing_pipeline(df: pd.DataFrame):
    feats, labels = pd.DataFrame(), pd.DataFrame()
    
    preprocess_data = Preprocessing_Dataset(df=df)
    
    dataloader_prep_data = DataLoader(preprocess_data,
                                      batch_size=64,
                                      num_workers=0,
                                      shuffle=False,
                                      collate_fn=custom_collate)

    for batch in dataloader_prep_data:
        temp_df = pd.DataFrame(batch)
        
        # Reading the signals
        temp_df = temp_df.merge(utils.read_sigs(temp_df))
        temp_df = temp_df.drop(columns=["extended filename"])
        temp_df = optimize(temp_df)
        
        # Extract features
        temp_feats_train = extract_feats.extract(temp_df)
        temp_feats_train = extract_feats.normalize(temp_feats_train)
        
        # Extract labels
        temp_labels_train = prep_labels.get_labels(temp_df)
        
        feats = pd.concat([feats, temp_feats_train], axis=0).reset_index(drop=True)
        labels = pd.concat([labels, temp_labels_train], axis=0).reset_index(drop=True)

    labels["labels"] = prep_labels.one_hot(labels["labels"])
    return feats, labels
    
# Optimizing the dataframe usage memory
# Credits to: https://medium.com/bigdatarepublic/advanced-pandas-optimize-speed-and-memory-a654b53be6c2
def optimize(df: pd.DataFrame):    
    def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
        floats = df.select_dtypes(include=["float64"]).columns.tolist()
        df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
        return df

    def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
        ints = df.select_dtypes(include=["int64"]).columns.tolist()
        df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
        return df

    def optimize_objects(df: pd.DataFrame) -> pd.DataFrame:
        df["label"] = df["label"].astype("category")
        return df

    return optimize_floats(optimize_ints(optimize_objects(df)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validation_dir",
        type=str,
        help="a path to a Kaldi-style data directory containting 'wav.scp', 'utt2spk' and 'segments'",
    )

    parser.add_argument(
        "--validation_split",
        type=float,
        help="a percetage of the training data to be used as a validation set, if an explicit validation \
                              set is not defined using -v",
    )

    parser.add_argument(
        "train_dir",
        type=str,
        help="a path to a Kaldi-style data directory containting 'wav.scp', 'utt2spk' and 'segments'",
    )

    parser.add_argument(
        "model_name",
        type=str,
        help="a filename for the model, the model will be saved as <model_name>.h5 in the output directory",
    )

    parser.add_argument(
        "out_dir",
        type=str,
        help="a path to an output directory where the model will be saved as <model_name>.pth",
    )

    parser.add_argument(
        "--binary_classification",
        action="store_true",
        help="use binary_classification (classes: speech and non-speech)",
    )

    args = parser.parse_args()

    if (
        os.path.exists(os.path.join(args.train_dir, "wav.scp"))
        and os.path.exists(os.path.join(args.train_dir, "segments"))
        and os.path.exists(os.path.join(args.train_dir, "utt2spk"))
    ) == False:
        utils.create_ava_files(
            path=args.train_dir, binary_classification=args.binary_classification
        )

    if args.validation_dir:
        if (
            os.path.exists(os.path.join(args.validation_dir, "wav.scp"))
            and os.path.exists(os.path.join(args.validation_dir, "segments"))
            and os.path.exists(os.path.join(args.validation_dir, "utt2spk"))
        ) == False:
            utils.create_ava_files(
                path=args.validation_dir,
                binary_classification=args.binary_classification,
            )
    
    # Preprocessing the train data
    feats_train, labels_train = pd.DataFrame(), pd.DataFrame()
        
    # Fetch the data
    data_train = prep_labels.prep_data(args.train_dir)
    
    feats_train, labels_train = preprocessing_pipeline(df=data_train)
    
    # Time distributing the data
    X = utils.time_distribute(np.vstack(feats_train["normalized-features"]), 15)
    y = utils.time_distribute(np.vstack(labels_train["labels"]), 15)

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    if not args.validation_dir:
        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, test_size=args.validation_split, random_state=seed
        )
    else:
        # Preprocessing the validation data
        feats_validation, labels_validation = pd.DataFrame(), pd.DataFrame()
        
        # Fetch the data
        data_validation = prep_labels.prep_data(args.validation_dir)
        
        feats_validation, labels_validation = preprocessing_pipeline(df=data_validation)

        X_validation = utils.time_distribute(
            np.vstack(feats_validation["normalized-features"]), 15
        )
        y_validation = utils.time_distribute(np.vstack(labels_validation["labels"]), 15)

        X_validation = X_validation.astype(np.float32)
        y_validation = y_validation.astype(np.float32)

    # Create the datasets and dataloaders
    training_dataset = AVA_Dataset(X=X_train, y=y_train)
    validation_dataset = AVA_Dataset(X=X_validation, y=y_validation)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    assert (
        y.shape[-1] == 2 or y.shape[-1] == 4
    ), f"ERROR: Number of classes {y.shape[-1]} is not equal to 2 or 4, see README for more info on using this training script."

    # Create the model
    epochs = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Voxseg(num_labels=y.shape[-1]).to(device)
    model.apply(weight_init)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-03, eps=1e-07)
    loss = nn.CrossEntropyLoss()
    save_best_model = SaveBestModel(output_dir=args.out_dir, model_name=args.model_name)
    os.makedirs(args.out_dir, exist_ok=True)
    training_log = pd.DataFrame()

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(
            device=device,
            train_loader=training_dataloader,
            optimizer=optimizer,
            epoch=epoch,
            model=model,
            loss=loss,
        )

        validation_loss, validation_acc = validation(
            device=device, validation_loader=validation_dataloader, model=model
        )

        save_best_model(
            current_valid_loss=validation_loss,
            current_valid_acc=validation_acc,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
        )

        training_log = pd.concat(
            [
                training_log,
                pd.DataFrame(
                    {
                        "epoch": [epoch],
                        "train_loss": [train_loss],
                        "train_accuracy": [train_acc],
                        "validation_loss": [validation_loss],
                        "validation_accuracy": [validation_acc],
                    }
                ),
            ],
            axis=0,
        )

    training_log.to_csv(
        os.path.join(args.out_dir, "training_log.csv"), index=False, sep=";"
    )
