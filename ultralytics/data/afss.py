import torch
import random
from tqdm import tqdm 
from ultralytics.utils.ops import xy
from datasets import ValDataset_for_DETR
from torch.utils.data import DataLoader

def calculate_iou_tensor(box1, box2):
    """计算单框对多框的IoU (xyxy格式)"""
    lt = torch.max(box1[:2], box2[:, :2])
    rb = torch.min(box1[2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter
    return inter / union

# 抗遗忘采样策略(Anti-Forgetting Sampling Strategy, AFSS)
class AFSSManager:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        # 初始化状态字典：{'P': float, 'R': float, 'ep': int}
        # 初始时, P和R都设为0 (全部视为 Hard, 强制学习)
        self.state_dict = {i: {'P': 0.0, 'R': 0.0, 'ep': -1} for i in range(num_samples)}
    
    # 根据当前状态, 生成本轮参与训练的图片索引, 每一个epoch都会更新生成一次omega, 用于训练的样本索引集, 
    # 这个方法并不更新state_dict中的P, R记录, 更新ep的记录, 用于下一次epoch的训练,
    # 在跨 5 epochs更新P,R之前, 其中的hard, moderate与easy样本的image_id均相同, 
    # 额外随机采样M1与E2使得中等样本与简单样本在每个epoch中都有机会被采样,
    # 同时, 为了保持样本的多样性, 每个epoch中, 中等样本与简单样本的采样数量不能超过40%的样本总数,
    def get_epoch_subset(self, current_epoch):
        # omega是用于训练的部分样本的索引集
        omega = set()
        # 存储img_i的3个难度等级列表
        easy_pool, moderate_pool, hard_pool = [], [], []
        
        # 1. 划分难度等级
        for img_id, state in self.state_dict.items():
            sufficiency = min(state['P'], state['R'])
            # 充分度划分: Easy, Moderate, Hard
            if sufficiency > 0.85:
                easy_pool.append(img_id)
            elif 0.55 <= sufficiency <= 0.85:
                moderate_pool.append(img_id)
            else:
                hard_pool.append(img_id)
        
        # 2. Hard 样本全采样
        omega.update(hard_pool)
        
        # 3. Moderate 样本短时覆盖 (距离上次超过3轮的强制加入)
        # current_epoch是当前训练轮次, 从0开始, state_dict 中的ep从-1开始所以需要减1
        forced_mod = [img for img in moderate_pool if current_epoch - 1 - self.state_dict[img]['ep'] >= 3]
        omega.update(forced_mod)
        
        # moderat额外采样数M1, 总共需要40%的中等难度样本训练, M1 = 40% 总量 - 已强制数量(遗忘掉了的moderate样本)
        # 这是一种优先覆盖遗忘样本, 再随机补充剩余样本
        remain_mod = list(set(moderate_pool) - set(forced_mod))
        M1 = max(0, int(0.4 * len(moderate_pool)) - len(forced_mod))
        if M1 > 0 and remain_mod:
            omega.update(random.sample(remain_mod, min(M1, len(remain_mod))))




