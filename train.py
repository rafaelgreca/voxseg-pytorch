import numpy as np
import argparse
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from voxseg.model import Voxseg, SaveBestModel
from voxseg.dataset import AVA_Dataset
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
torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)


def train(device, train_loader, optimizer, epoch, model) -> Tuple[float, float]:
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

        l = F.binary_cross_entropy(output, target)
        l.backward()
        optimizer.step()

        training_loss += l.item()

        pred = output.argmax(dim=2)
        training_acc += pred.eq(target.argmax(dim=2)).sum().item()

        pctg_batch = 100.0 * batch_idx / len(train_loader)

        if round(pctg_batch, 0) in pctg_to_print:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                )
            )
            pctg_to_print.remove(pctg_batch)

    training_loss /= len(train_loader.dataset) / 64
    training_acc /= len(train_loader.dataset) * 15
    return training_loss, training_acc


def validation(device, validation_loader, model) -> Tuple[float, float]:
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

            l = F.binary_cross_entropy(output, target)
            validation_loss += l.item()

            pred = output.argmax(dim=2)
            validation_acc += pred.eq(target.argmax(dim=2)).sum().item()

    validation_loss /= len(validation_loader.dataset) / 64
    validation_acc /= len(validation_loader.dataset) * 15
    return validation_loss, validation_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        os.path.exists(os.path.join(args.train_dir, "wav.scp"))
        and os.path.exists(os.path.join(args.train_dir, "segments"))
        and os.path.exists(os.path.join(args.train_dir, "utt2spk"))
    ) == False:
        utils.create_ava_files(args.train_dir)

    if args.validation_dir:
        if (
            os.path.exists(os.path.join(args.validation_dir, "wav.scp"))
            and os.path.exists(os.path.join(args.validation_dir, "segments"))
            and os.path.exists(os.path.join(args.validation_dir, "utt2spk"))
        ) == False:
            utils.create_ava_files(args.validation_dir)

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
            X, y, test_size=args.validation_split, random_state=seed
        )

    # Create the datasets and dataloaders
    training_dataset = AVA_Dataset(X=X_train, y=y_train)
    validation_dataset = AVA_Dataset(X=X_validation, y=y_validation)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=64,
        num_workers=1,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=64,
        num_workers=1,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    assert (
        y.shape[-1] == 2 or y.shape[-1] == 4
    ), f"ERROR: Number of classes {y.shape[-1]} is not equal to 2 or 4, see README for more info on using this training script."

    # Create the model
    epochs = 25
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model = Voxseg(num_labels=y.shape[-1]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-03, eps=1e-07)
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
