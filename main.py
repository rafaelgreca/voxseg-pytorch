import argparse
import os
import torch
from voxseg import extract_feats, run_cnnlstm, utils, evaluate, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Extracts features and run VAD to generate endpoints.",
    )
    parser.add_argument(
        "-M",
        "--model_path",
        type=str,
        help="a path to a trained vad model saved as in .h5 format, overrides default pretrained model",
    )
    parser.add_argument(
        "-s",
        "--speech_thresh",
        type=float,
        help="a decision threshold value between (0,1) for speech vs non-speech, defaults to 0.5",
    )
    parser.add_argument(
        "-m",
        "--speech_w_music_thresh",
        type=float,
        help="a decision threshold value between (0,1) for speech_with_music vs non-speech, defaults to 0.5, \
                       increasing will remove more speech_with_music, useful for downsteam ASR",
    )
    parser.add_argument(
        "-f",
        "--median_filter_kernel",
        type=int,
        help="a kernel size for a median filter to smooth the output labels, defaults to 1 (no smoothing)",
    )
    parser.add_argument(
        "-e",
        "--eval_dir",
        type=str,
        help="a path to a Kaldi-style data directory containing the ground truth VAD segments for evaluation",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="a path to a Kaldi-style data directory containting 'wav.scp', and optionally 'segments'",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="a path to an output directory where the output segments will be saved",
    )
    parser.add_argument(
        "--binary_classification",
        action="store_true",
        help="use binary_classification (classes: speech and non-speech)",
    )
    args = parser.parse_args()

    if (
        os.path.exists(os.path.join(args.data_dir, "wav.scp"))
        and os.path.exists(os.path.join(args.data_dir, "segments"))
        and os.path.exists(os.path.join(args.data_dir, "utt2spk"))
    ) == False:
        utils.create_ava_files(args.data_dir)

    data = extract_feats.prep_data(args.data_dir)
    feats = extract_feats.extract(data)
    feats = extract_feats.normalize(feats)

    if args.binary_classification:
        model = model.Voxseg(num_labels=2)
    else:
        model = model.Voxseg(num_labels=4)

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    else:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    os.getcwd(),
                    "checkpoints",
                    "model.pth",
                )
            )["model_state_dict"]
        )
    if args.speech_thresh is not None:
        speech_thresh = args.speech_thresh
    else:
        speech_thresh = 0.5
    if args.speech_w_music_thresh is not None:
        speech_w_music_thresh = args.speech_w_music_thresh
    else:
        speech_w_music_thresh = 0.5
    if args.median_filter_kernel is not None:
        filt = args.median_filter_kernel
    else:
        filt = 1

    model.eval()
    targets = run_cnnlstm.predict_targets(model, feats)
    endpoints = run_cnnlstm.decode(targets, speech_thresh, speech_w_music_thresh, filt)
    run_cnnlstm.to_data_dir(endpoints, args.out_dir)

    if args.eval_dir is not None:
        wav_scp, wav_segs, _ = utils.process_data_dir(args.data_dir)
        _, sys_segs, _ = utils.process_data_dir(args.out_dir)
        _, ref_segs, _ = utils.process_data_dir(args.eval_dir)
        scores = evaluate.score(wav_scp, sys_segs, ref_segs, wav_segs)
        print(scores)
        evaluate.print_confusion_matrix(scores)
