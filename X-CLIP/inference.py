import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip

from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download
from transformers import VideoMAEFeatureExtractor

from clip.clip import tokenize


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config): 
    # train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    # text_labels = generate_text(train_data)

    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)

    # load weights
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    load_state_dict = checkpoint['model']
    msg = model.load_state_dict(load_state_dict, strict=False)
    logger.info(f"resume model: {msg}")

    # load video + several texts
    # video clip consists of 300 frames (10 seconds at 30 FPS)
    file_path = hf_hub_download(
        repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
    )
    vr = VideoReader(file_path, num_threads=1, ctx=cpu(0))

    # sample 8 frames
    vr.seek(0)
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=len(vr))
    buffer = vr.get_batch(indices).asnumpy()

    # create a list of NumPy arrays
    video = [buffer[i] for i in range(buffer.shape[0])]

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    inputs = feature_extractor(video, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    print("Shape of pixel values:", pixel_values.shape)

    text_labels = tokenize(["playing sports", "eating spaghetti", "go shopping"])

    # inference
    model.eval()
    with torch.no_grad():
        logits = model(image=pixel_values, text=text_labels)
        probs = logits.softmax(dim=1)
        print("Probs:", probs)


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # seed
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    # save config 
    logger.info(config)
    shutil.copy(args.config, config.OUTPUT)

    main(config)