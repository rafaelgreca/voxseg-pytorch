# Script for training custom VAD model for the voxseg toolkit
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from voxseg.model import Voxseg, SaveBestModel
from voxseg.dataset import AVA_Dataset
from voxseg import utils, extract_feats, prep_labels
from torch.utils.data import DataLoader


def train(device, train_loader, optimizer, epoch, loss):
    model.train()
    training_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        data = batch["X"]
        data = data.unsqueeze(2)
        target = batch["y"]

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        l = loss(output, target)
        l.backward()
        optimizer.step()

        training_loss += l.item()

        if batch_idx % 5 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                )
            )

    training_loss /= len(train_loader.dataset)
    return training_loss


def validation(device, validation_loader, loss):
    model.eval()
    validation_loss = 0

    with torch.no_grad():
        for batch in validation_loader:
            data = batch["X"]
            data = data.unsqueeze(2)
            target = batch["y"]

            data, target = data.to(device), target.to(device)
            output = model(data)

            l = loss(output, target)
            validation_loss += l.item()

    validation_loss /= len(validation_loader.dataset)
    return validation_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CHPC_VAD_train.py",
        description="Train an instance of the voxseg VAD model.",
    )

    parser.add_argument(
        "-v",
        "--validation_dir",
        type=str,
        help="a path to a Kaldi-style data directory containting 'wav.scp', 'utt2spk' and 'segments'",
    )

    parser.add_argument(
        "-validation_split",
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
        help="a path to an output directory where the model will be saved as <model_name>.h5",
    )

    args = parser.parse_args()

    if (
        os.path.exists(os.path.join(os.getcwd(), "wav.scp"))
        and os.path.exists(os.path.join(os.getcwd(), "segments"))
        and os.path.exists(os.path.join(os.getcwd(), "utt2spk"))
    ) == False:

        utils.create_ava_files(args.train_dir)

    # Fetch data
    data_train = prep_labels.prep_data(args.train_dir)

    if args.validation_dir:
        data_dev = prep_labels.prep_data(args.validation_dir)

    # Extract features
    feats_train = extract_feats.extract(data_train)
    feats_train = extract_feats.normalize(feats_train)

    if args.validation_dir:
        feats_dev = extract_feats.extract(data_dev)
        feats_dev = extract_feats.normalize(feats_dev)

    # Extract labels
    labels_train = prep_labels.get_labels(data_train)
    labels_train["labels"] = prep_labels.one_hot(labels_train["labels"])

    if args.validation_dir:
        labels_dev = prep_labels.get_labels(data_dev)
        labels_dev["labels"] = prep_labels.one_hot(labels_dev["labels"])

    # Train model
    X = utils.time_distribute(np.vstack(feats_train["normalized-features"]), 15)
    y = utils.time_distribute(np.vstack(labels_train["labels"]), 15)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if args.validation_dir:
        X_validation = utils.time_distribute(
            np.vstack(feats_dev["normalized-features"]), 15
        )
        y_validation = utils.time_distribute(np.vstack(labels_dev["labels"]), 15)

        X_validation = X_validation.astype(np.float32)
        y_validation = y_validation.astype(np.float32)

    if not args.validation_dir:
        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, test_size=args.validation_split, random_state=0
        )

    training_dataset = AVA_Dataset(X=X_train, y=y_train)
    validation_dataset = AVA_Dataset(X=X_validation, y=y_validation)

    training_dataloader = DataLoader(
        training_dataset, batch_size=64, num_workers=0, shuffle=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=64, num_workers=0, shuffle=True
    )

    epochs = 10
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model = Voxseg(num_labels=2).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()
    save_best_model = SaveBestModel(output_dir=args.out_dir, model_name=args.model_name)
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train(
            device=device,
            train_loader=training_dataloader,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
        )

        validation_loss = validation(
            device=device, validation_loader=validation_dataloader, loss=loss
        )

        save_best_model(
            current_valid_loss=validation_loss,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            criterion=loss,
        )
