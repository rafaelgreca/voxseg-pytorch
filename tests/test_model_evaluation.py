import os
import pytest
import pandas as pd
import torch
from voxseg import evaluate, extract_feats, run_cnnlstm, utils
from voxseg.model import TimeDistributed, Voxseg


def test1():
    """
    Testing features extraction and normalization step
    """
    data = extract_feats.prep_data(os.path.join(os.getcwd(), "tests", "files", "data"))
    feats = extract_feats.extract(data)
    feats = extract_feats.normalize(feats)
    utils.save(
        feats, os.path.join(os.getcwd(), "tests", "files", "features", "feats.h5")
    )

    assert not feats is None
    assert "normalized-features" in feats.columns.tolist()
    assert len(feats["normalized-features"].values.tolist()) > 0


def test2():
    """
    Testing model prediction step
    """

    ## FIXME: Check if targets size before initiate the model
    model = Voxseg(2)
    model.load_state_dict(
        torch.load(os.path.join(os.getcwd(), "checkpoints", "model.pth"))[
            "model_state_dict"
        ]
    )
    model.eval()

    feats = pd.read_hdf(
        os.path.join(os.getcwd(), "tests", "files", "features", "feats.h5")
    )

    assert not feats is None

    targets = run_cnnlstm.predict_targets(model, feats)

    assert not targets is None
    assert "predicted-targets" in targets.columns.tolist()
    assert len(targets["predicted-targets"].values.tolist()) > 0

    endpoints = run_cnnlstm.decode(targets)

    assert not endpoints is None
    assert ("start" in endpoints.columns.tolist()) and (
        "end" in endpoints.columns.tolist()
    )
    assert (len(endpoints["start"].values.tolist()) > 0) and (
        len(endpoints["end"].values.tolist()) > 0
    )

    run_cnnlstm.to_data_dir(
        endpoints, os.path.join(os.getcwd(), "tests", "files", "output")
    )

    assert os.path.exists(
        os.path.join(os.getcwd(), "tests", "files", "output", "segments")
    )


def test3():
    """
    Testing the model evaluation step
    """
    wav_scp, wav_segs, _ = utils.process_data_dir(
        os.path.join(os.getcwd(), "tests", "files", "data")
    )
    _, sys_segs, _ = utils.process_data_dir(
        os.path.join(os.getcwd(), "tests", "files", "output")
    )
    _, ref_segs, _ = utils.process_data_dir(
        os.path.join(os.getcwd(), "tests", "files", "ground_truth")
    )

    assert not wav_scp is None
    assert (not sys_segs is None) and (not ref_segs is None)

    scores = evaluate.score(wav_scp, sys_segs, ref_segs, wav_segs)
    # evaluate.print_confusion_matrix(scores)

    assert not scores is None
