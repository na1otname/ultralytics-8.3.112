import torch
import random
from tqdm import tqdm 
from ultralytics.utils.ops import xy
from datasets import ValDataset_for_DETR
from torch.utils.data import DataLoader

def calculate_iou_tensor(box1, box2):
    """计算单框对多框的IoU (xyxy格式)"""
    lt = torch.max(box1[:2], box2[:, :2])