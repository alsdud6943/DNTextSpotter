# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import gc
import torch

# Register custom backbones so the registry sees them
import adet.modeling.vitae_v2
import adet.modeling.swin

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

import sys
import os

# Add the current project directory to the front of sys.path to ensure we import the correct modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.insert(0, project_dir)

from adet.config import get_cfg
from demo.predictor import VisualizationDemo
from adet.utils.lexicon_ctc_custom import LexiconConstrainedCTC

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
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    # Load default checkpoint in code (override CLI)
    cfg.MODEL.WEIGHTS = "/home/hanta/minyeong/DNTextSpotter/vitaev2_pretrain_tt_model_final.pth"
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file",)
    parser.add_argument("--input", nargs="+", help="A list of space separated input images; ""or a single glob pattern such as 'directory/*.jpg'",)
    parser.add_argument("--output", help="A directory to save output visualizations (overrides config file setting).")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="Minimum score for instance predictions to be shown",)
    # Lexicon arguments moved to YAML config file
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER,)
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # Setup lexicon-constrained CTC decoder
    lexicon_decoder = None
    
    # Get lexicon settings from config (with defaults if not present)
    lexicon_type = getattr(cfg, "LEXICON", {}).get("TYPE", "none")
    lexicon_file = getattr(cfg, "LEXICON", {}).get("FILE", None)
    lexicon_threshold = getattr(cfg, "LEXICON", {}).get("THRESHOLD", 0.6)
    
    if lexicon_type != "none":
        # For the pretrained model, force voc_size to 37 since that's what it was trained with
        # regardless of what the config says
        voc_size = 37
        logger.info(f"Using voc_size={voc_size} (forced for pretrained model compatibility)")
        
        # Create vocabulary based on lexicon type
        if lexicon_type == "custom":
            if not lexicon_file:
                logger.error("Custom lexicon type requires LEXICON.FILE in config")
                exit(1)
            try:
                with open(lexicon_file, 'r', encoding='utf-8') as f:
                    vocabulary = [line.strip().upper() for line in f if line.strip()]
                logger.info(f"Loaded custom lexicon with {len(vocabulary)} words from {lexicon_file}")
            except Exception as e:
                logger.error(f"Failed to load custom lexicon file: {e}")
                exit(1)
        else:
            vocabulary = []
            
        lexicon_decoder = LexiconConstrainedCTC(
            vocabulary=vocabulary,
            voc_size=voc_size,
            similarity_threshold=lexicon_threshold
        )
        logger.info(f"Lexicon-constrained CTC decoder initialized with threshold {lexicon_threshold}")
    else:
        logger.info("Using standard CTC decoding (no lexicon constraint)")

    demo = VisualizationDemo(cfg)
    
    # Use input glob from config if no CLI --input provided
    if not args.input and getattr(cfg, "INFERENCE", None) and cfg.INFERENCE.INPUT_GLOB:
        args.input = glob.glob(cfg.INFERENCE.INPUT_GLOB)
    
    # Use output directory from config if no CLI --output provided
    if not args.output and getattr(cfg, "INFERENCE", None) and cfg.INFERENCE.OUTPUT_DIR:
        args.output = cfg.INFERENCE.OUTPUT_DIR
        logger.info(f"Using output directory from config: {args.output}")

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
            
            # Debug: print what's in predictions
            if "instances" in predictions:
                instances = predictions["instances"]
                print(f"Number of instances: {len(instances)}")
                print(f"Instance fields: {instances.get_fields().keys()}")
                
                ######## Print detected text if available ########
                if hasattr(instances, 'recs') and len(instances) > 0:
                    print("Detected text:")
                    
                    for i, rec in enumerate(instances.recs):
                        if lexicon_decoder is not None:
                            # Use lexicon-constrained decoding
                            standard_text = lexicon_decoder.standard_ctc_decode(rec)
                            constrained_text = lexicon_decoder.lexicon_constrained_decode(rec)
                            print(f"  {i}: Standard: '{standard_text}' -> Constrained: '{constrained_text}'")
                            if standard_text != constrained_text:
                                similarity = lexicon_decoder.compute_similarity(standard_text, constrained_text)
                                print(f"      Correction applied (similarity: {similarity:.3f})")
                        else:
                            # Use standard CTC decoding
                            CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
                            text = ""
                            last_char = -1
                            for c in rec:
                                c = int(c)
                                # Skip blank tokens (typically the highest index, here 37)
                                if c < len(CTLABELS) and c != last_char:
                                    text += CTLABELS[c]
                                last_char = c
                            print(f"  {i}: '{text}'")
                        
                        print(f"      Raw indices: {rec.tolist()}")
            
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
                # Save visualization if available
                out_filename = os.path.join(args.output, os.path.basename(path))
                if visualized_output is not None:
                    visualized_output.save(out_filename)
                    logger.info(f"Visualization saved to: {out_filename}")
                else:
                    logger.warning(f"No visualization generated for {path}")                

                ########## Save transcription results to text file ##########
                if "instances" in predictions and hasattr(predictions["instances"], 'recs') and len(predictions["instances"]) > 0:
                    # Create text filename by replacing image extension with .txt
                    base_name = os.path.splitext(os.path.basename(path))[0]
                    text_filename = os.path.join(args.output, f"{base_name}_transcription.txt")
                    
                    with open(text_filename, 'w', encoding='utf-8') as f:
                        instances = predictions["instances"]
                        
                        for i, rec in enumerate(instances.recs):
                            if lexicon_decoder is not None:
                                # Use lexicon-constrained decoding
                                standard_text = lexicon_decoder.standard_ctc_decode(rec)
                                constrained_text = lexicon_decoder.lexicon_constrained_decode(rec)
                                f.write(f"Standard: '{standard_text}' -> Constrained: '{constrained_text}' (score: {instances.scores[i]:.3f})\n")
                            else:
                                # Use standard CTC decoding
                                CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
                                text = ""
                                last_char = -1
                                for c in rec:
                                    c = int(c)
                                    if c < len(CTLABELS) and c != last_char:
                                        text += CTLABELS[c]
                                    last_char = c
                                f.write(f"'{text}' (score: {instances.scores[i]:.3f})\n")
                    
                    logger.info(f"Transcription saved to: {text_filename}")
                else:
                    logger.info(f"No text detected in {path}")

            else:
                if visualized_output is not None:
                    cv2.imshow("Prediction", visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
                else:
                    logger.warning(f"No visualization generated for {path}")
    
            # Clean up memory after each image to prevent segmentation faults
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    else:
        logger.error("No input images provided. Please use the --input argument.")
