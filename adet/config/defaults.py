from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- # 
# Additional Configs
_C.MODEL.USEDN = True

# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = False
_C.INPUT.CROP.CROP_INSTANCE = True
_C.INPUT.ROTATE = True

_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3

_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = "conv"
_C.MODEL.TOP_MODULE.DIM = 16


# ---------------------------------------------------------------------------- #
# BAText Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BATEXT = CN()
_C.MODEL.BATEXT.VOC_SIZE = 96
_C.MODEL.BATEXT.NUM_CHARS = 25
_C.MODEL.BATEXT.POOLER_RESOLUTION = (8, 32)
_C.MODEL.BATEXT.IN_FEATURES = ["p2", "p3", "p4"]
_C.MODEL.BATEXT.POOLER_SCALES = (0.25, 0.125, 0.0625)
_C.MODEL.BATEXT.SAMPLING_RATIO = 1
_C.MODEL.BATEXT.CONV_DIM = 256
_C.MODEL.BATEXT.NUM_CONV = 2
_C.MODEL.BATEXT.RECOGNITION_LOSS = "ctc"
_C.MODEL.BATEXT.RECOGNIZER = "attn"
_C.MODEL.BATEXT.CANONICAL_SIZE = 96  # largest min_size for level 3 (stride=8)
_C.MODEL.BATEXT.USE_COORDCONV = False
_C.MODEL.BATEXT.USE_AET = False
_C.MODEL.BATEXT.CUSTOM_DICT = "" # Path to the class file.


# ---------------------------------------------------------------------------- #
# SwinTransformer Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.TYPE = 'tiny'
_C.MODEL.SWIN.DROP_PATH_RATE = 0.2

# ---------------------------------------------------------------------------- #
# ViTAE-v2 Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ViTAEv2 = CN()
_C.MODEL.ViTAEv2.TYPE = 'vitaev2_s'
_C.MODEL.ViTAEv2.DROP_PATH_RATE = 0.2

# ---------------------------------------------------------------------------- #
# (Deformable) Transformer Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.ENABLED = False
_C.MODEL.TRANSFORMER.INFERENCE_TH_TEST = 0.4 
_C.MODEL.TRANSFORMER.AUX_LOSS = True
_C.MODEL.TRANSFORMER.ENC_LAYERS = 6
_C.MODEL.TRANSFORMER.DEC_LAYERS = 6 
_C.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 1024
_C.MODEL.TRANSFORMER.HIDDEN_DIM = 256 
_C.MODEL.TRANSFORMER.DROPOUT = 0.0
_C.MODEL.TRANSFORMER.NHEADS = 8
_C.MODEL.TRANSFORMER.NUM_QUERIES = 100
_C.MODEL.TRANSFORMER.ENC_N_POINTS = 4
_C.MODEL.TRANSFORMER.DEC_N_POINTS = 4
_C.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE = 6.283185307179586  # 2 PI
_C.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS = 4
_C.MODEL.TRANSFORMER.VOC_SIZE = 37  # a-z + 0-9 + unknown
_C.MODEL.TRANSFORMER.NUM_POINTS = 25  # the number of point queries for each instance
_C.MODEL.TRANSFORMER.TEMPERATURE = 10000
_C.MODEL.TRANSFORMER.BOUNDARY_HEAD = True # True: with boundary predictions, False: only with center lines
_C.MODEL.TRANSFORMER.SCALAR = 5
_C.MODEL.TRANSFORMER.CLASS_NOISE_SCALE = 0
_C.MODEL.TRANSFORMER.TEXTS_NOISE_SCALE = 0.2
_C.MODEL.TRANSFORMER.CTRL_POINTS_NOISE_SCALE = 0.4
_C.MODEL.TRANSFORMER.NUM_PATTERNS = 0
_C.MODEL.TRANSFORMER.CONTRASTIVE = True



_C.MODEL.TRANSFORMER.LOSS = CN()
_C.MODEL.TRANSFORMER.LOSS.AUX_LOSS = True
_C.MODEL.TRANSFORMER.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.TRANSFORMER.LOSS.FOCAL_GAMMA = 2.0
# bezier proposal loss
_C.MODEL.TRANSFORMER.LOSS.BEZIER_CLASS_WEIGHT = 1.0
_C.MODEL.TRANSFORMER.LOSS.BEZIER_COORD_WEIGHT = 1.0
_C.MODEL.TRANSFORMER.LOSS.BEZIER_SAMPLE_POINTS = 25
# supervise the sampled on-curve points but not 4 Bezier control points

# target loss
_C.MODEL.TRANSFORMER.LOSS.POINT_CLASS_WEIGHT = 1.0
_C.MODEL.TRANSFORMER.LOSS.POINT_CLASS_WEIGHT = 1.0
_C.MODEL.TRANSFORMER.LOSS.POINT_COORD_WEIGHT = 1.0
_C.MODEL.TRANSFORMER.LOSS.POINT_TEXT_WEIGHT = 0.5
_C.MODEL.TRANSFORMER.LOSS.BOUNDARY_WEIGHT = 0.5

_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.LR_BACKBONE = 1e-5
_C.SOLVER.LR_BACKBONE_NAMES = []
_C.SOLVER.LR_LINEAR_PROJ_NAMES = []
_C.SOLVER.LR_LINEAR_PROJ_MULT = 0.1

# 1 - Generic, 2 - Weak, 3 - Strong (for icdar2015)
# 1 - Full lexicon (for totaltext)
_C.TEST.LEXICON_TYPE = 1
# ---------------------------------------------------------------------------- #
# Inference Options
# ---------------------------------------------------------------------------- #
_C.INFERENCE = CN()
_C.INFERENCE.INPUT_GLOB = ""  # Glob or directory path for inference images
_C.INFERENCE.OUTPUT_DIR = ""  # Directory to save output visualizations and transcriptions

# ---------------------------------------------------------------------------- #
# Lexicon-constrained CTC Options
# ---------------------------------------------------------------------------- #
_C.LEXICON = CN()
_C.LEXICON.TYPE = "none"  # Type: "none" (standard CTC) or "custom" (from file)
_C.LEXICON.FILE = ""  # Path to custom lexicon file (one word per line)
_C.LEXICON.THRESHOLD = 0.6  # Similarity threshold for lexicon matching (0.0-1.0)