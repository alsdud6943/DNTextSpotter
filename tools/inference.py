# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from adet.config import get_cfg
from demo.predictor import VisualizationDemo

def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    # Load default checkpoint in code (override CLI)
    cfg.MODEL.WEIGHTS = "/home/hanta/minyeong/DNTextSpotter/vitaev2_pretrain_tt_model_final.pth"
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file",)
    parser.add_argument("--input", nargs="+", help="A list of space separated input images; ""or a single glob pattern such as 'directory/*.jpg'",)
    parser.add_argument("--output", help="A directory to save output visualizations.")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="Minimum score for instance predictions to be shown",)
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER,)
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    # Use input glob from config if no CLI --input provided
    if not args.input and getattr(cfg, "INFERENCE", None) and cfg.INFERENCE.INPUT_GLOB:
        args.input = glob.glob(cfg.INFERENCE.INPUT_GLOB)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        
        if args.output:
            os.makedirs(args.output, exist_ok=True)

        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            try:
                img = read_image(path, format="BGR")
            except Exception as e:
                logger.warning(f"Failed to read image {path}: {e}")
                continue
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                out_filename = os.path.join(args.output, os.path.basename(path))
                visualized_output.save(out_filename)
            else:
                cv2.imshow("Prediction", visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    else:
        logger.error("No input images provided. Please use the --input argument.")
