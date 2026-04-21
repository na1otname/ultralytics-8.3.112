import torch
import random
from tqdm import tqdm 
from ultralytics.utils.ops import xy
from ultralytics.data import build_dataloader, build_yolo_dataset, converter
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
        
        # 4. Easy 样本持续复习 (距离上次超过10轮的强制加入)
        forced_easy = [img for img in easy_pool if current_epoch - 1 - self.state_dict[img]['ep'] >= 10]
        # 强制加入的数量最多只能占 easy 样本的 1%(0.5 × 2%)
        max_forced_easy = int(0.5 * 0.02 * len(easy_pool))
        if len(forced_easy) > max_forced_easy:
            forced_easy = random.sample(forced_easy, max_forced_easy)
        omega.update(forced_easy)
        
        # easy额外采样数E2, 总共只希望 2% 的easy样本参与训练, E2 = 2% 总量 - 已强制数量(遗忘掉了的easy样本, 最多只有1%)
        remain_easy = list(set(easy_pool) - set(forced_easy))
        E2 = max(0, int(0.02 * len(easy_pool)) - len(forced_easy))
        if E2 > 0 and remain_easy:
            omega.update(random.sample(remain_easy, min(E2, len(remain_easy))))
        
        # 5. 更新参与本轮训练的样本的记录
        # omega是用于强制完整训练的训练集样本, 需要被更新到current_epoch
        for img_id in omega:
            self.state_dict[img_id]['ep'] = current_epoch
            
        return list(omega)
    
    def print_sufficiency_distribution(self):
        """统计并打印当前数据集的学习充分度分布 (Easy/Moderate/Hard)"""
        easy_count, mod_count, hard_count = 0, 0, 0
        for state in self.state_dict.values():
            suff = min(state['P'], state['R'])
            if suff > 0.85:
                easy_count += 1
            elif 0.55 <= suff <= 0.85:
                mod_count += 1
            else:
                hard_count += 1
        
        total = len(self.state_dict)
        if total == 0:
            print("  AFSS State Dict is empty!")
            return
        
        print(f"  -> Easy     (>0.85): {easy_count:5d} / {total} ({easy_count/total*100:.1f}%)")
        print(f"  -> Moderate (0.55-0.85): {mod_count:5d} / {total} ({mod_count/total*100:.1f}%)")
        print(f"  -> Hard     (<0.55): {hard_count:5d} / {total} ({hard_count/total*100:.1f}%)")
        print("-" * 60)

    # 利用验证集的无增强DataLoader对全体训练集做一次快速推断, 更新state_dict中的P, R记录, 跨5个epochs更新一次
    # 全体训练集边推理边计算, 效率不高, 但可以实时监控模型在训练集上的学习进度, 所以设置5epochs做一次更新
    @torch.no_grad()
    def evaluate_and_update(self, model, dataloader, conf_thresh=0.2, iou_thresh=0.5):
        model.eval()
        
        for batch in dataloader:
            imgs = batch['img'].to(model.device)
            gt_boxes = batch['bboxes']  # ground truth
            gt_cls = batch['cls']
            batch_idx = batch['batch_idx']
            
            # 模型推理
            outputs = model(imgs)
            
            # 对每个图像计算P和R
            for img_id in unique(batch_idx):
                pred_boxes = outputs[img_id].boxes
                gt_mask = batch_idx == img_id
                gt_boxes_img = gt_boxes[gt_mask]
                
                # 计算Precision和Recall
                P, R = self._calculate_pr(pred_boxes, gt_boxes_img, iou_thresh, conf_thresh)
                
                # 更新状态
                self.state_dict[img_id]['P'] = P
                self.state_dict[img_id]['R'] = R
        
        model.train()





