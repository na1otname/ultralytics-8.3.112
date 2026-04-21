import torch
import random
from torch.utils.data import Sampler
from collections import defaultdict
from ultralytics.utils import LOGGER


def iou_xyxy(box1, box2):
    """计算单框对多框的IoU (xyxy格式)"""
    lt = torch.max(box1[:2], box2[:, :2])
    rb = torch.min(box1[2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)

# 抗遗忘采样策略(Anti-Forgetting Sampling Strategy, AFSS)
class AFSSManager:
    """
        AFSS Manager: track per-image P,R and epoch, provide epoch subset and periodic evaluation update.
        state_dict: {img_idx: {'P': float, 'R': float, 'ep': int}}
    """
    def __init__(self, num_samples, easy_frac=0.02, moderate_frac=0.4, forced_mod_gap=3, forced_easy_gap=10):
        self.num_samples = int(num_samples)
        self.easy_frac = easy_frac
        self.moderate_frac = moderate_frac
        self.forced_mod_gap = forced_mod_gap
        self.forced_easy_gap = forced_easy_gap
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
        forced_mod = [img for img in moderate_pool if current_epoch - 1 - self.state_dict[img]['ep'] >= self.forced_mod_gap]
        omega.update(forced_mod)
        
        # moderat额外采样数M1, 总共需要40%的中等难度样本训练, M1 = 40% 总量 - 已强制数量(遗忘掉了的moderate样本)
        # 这是一种优先覆盖遗忘样本, 再随机补充剩余样本
        remain_mod = list(set(moderate_pool) - set(forced_mod))
        M1 = max(0, int(0.4 * len(moderate_pool)) - len(forced_mod))
        if M1 > 0 and remain_mod:
            omega.update(random.sample(remain_mod, min(M1, len(remain_mod))))
        
        # 4. Easy 样本持续复习 (距离上次超过10轮的强制加入)
        forced_easy = [img for img in easy_pool if current_epoch - 1 - self.state_dict[img]['ep'] >= self.forced_easy_gap]
        max_forced_easy = int(0.5 * self.easy_frac * len(easy_pool))

        if len(forced_easy) > max_forced_easy and max_forced_easy > 0:
            forced_easy = random.sample(forced_easy, max_forced_easy)
        omega.update(forced_easy)
        
        # easy额外采样数E2, 总共只希望 2% 的easy样本参与训练, E2 = 2% 总量 - 已强制数量(遗忘掉了的easy样本, 最多只有1%)
        remain_easy = list(set(easy_pool) - set(forced_easy))
        E2 = max(0, int(self.easy_frac * len(easy_pool)) - len(forced_easy))
        if E2 > 0 and remain_easy:
            omega.update(random.sample(remain_easy, min(E2, len(remain_easy))))
        
        # 5. 更新参与本轮训练的样本的记录
        # omega是用于强制完整训练的训练集样本, 需要被更新到current_epoch
        for img_id in omega:
            self.state_dict[img_id]['ep'] = current_epoch
        return list(omega)
    
    def print_sufficiency_distribution(self):
        """统计并打印当前数据集的学习充分度分布 (Easy/Moderate/Hard)"""
        easy, mod, hard = 0, 0, 0
        for s in self.state_dict.values():
            suff = min(s['P'], s['R'])
            if suff > 0.85:
                easy += 1
            elif 0.55 <= suff <= 0.85:
                mod += 1
            else:
                hard += 1
        total = self.num_samples
        if total == 0:
            LOGGER.info("AFSS state empty")
            return
        LOGGER.info(f"AFSS sufficiency: Easy {easy}/{total}, Moderate {mod}/{total}, Hard {hard}/{total}")


    # 利用验证集的无增强DataLoader对全体训练集做一次快速推断, 更新state_dict中的P, R记录, 跨5个epochs更新一次
    # 全体训练集边推理边计算, 效率不高, 但可以实时监控模型在训练集上的学习进度, 所以设置5epochs做一次更新
    @torch.no_grad()
    def evaluate_and_update(self, model, dataloader, conf_thresh=0.2, iou_thresh=0.5, device=None):
        """
        Run inference over dataloader (no augmentation) and update self.state_dict P,R per image index.
        Assumptions:
          - dataloader yields batch dictionaries as in YOLODataset.collate_fn
          - batch['batch_idx'] contains per-instance image index (as in YOLO collate)
          - model(...) returns ultralytics Results-like object for each image in batch
        NOTE: You may need to adapt parsing if your model returns different structure.
        """        
        model.eval()
        dev = device if device is not None else getattr(model, "device", None)
        
        for batch in dataloader:
            imgs = batch['img'].to(dev)
            gt_boxes = batch['bboxes']  # ground truth
            batch_idx = batch['batch_idx'] # tensor of sample-image-indices concatenated for batch
            # Model forward: may return list of Results (per image) or a batched result
            preds = model(imgs)
            # Normalize preds to list-of-results: try to handle common cases
            # If preds is Results or list-like with .boxes, adapt accordingly
            # We assume preds is iterable and preds[i] corresponds to i-th image in batch (or use model(..., stream=True))
            # Extract per-sample predictions:
            per_image_preds = []
            # If model returns a single Results for whole batch with attribute .boxes for each image, try to split:
            if isinstance(preds, (list, tuple)):
                per_image_preds = preds
            else:
                # Try to call model._split if exists else treat as single
                try:
                    per_image_preds = list(preds)
                except Exception:
                    per_image_preds = [preds]

            # Now for each image in the batch compute P,R
            # Get unique image ids in this mini-batch
            unique_ids = torch.unique(batch_idx)
            for img_id in unique_ids.tolist():
                # indices for this image in concatenated targets
                mask = batch_idx == img_id
                gt_boxes = batch['bboxes'][mask].cpu()  # (M,4) in xywh norm format
                # Convert gt xywh -> xyxy in pixel normalized space (dataset stores normalized)
                if gt_boxes.shape[0] == 0:
                    # no GT boxes: treat as easy with P=1,R=1? Here set P=1,R=1 to not mark as hard
                    self.state_dict[img_id]['P'] = 1.0
                    self.state_dict[img_id]['R'] = 1.0
                    continue
                # Determine which prediction corresponds to this image in per_image_preds:
                # trying to index by order: assume per_image_preds order matches dataloader batch image order
                # fallback: compute mapping by batch ordering
                # Here we attempt to find prediction by sequential index:
                try:
                    # compute local index (first occurrence position)
                    local_pos = (batch_idx == img_id).nonzero(as_tuple=False)[0].item()
                    # determine which per_image_preds index: approximate by floor(local_pos / imgs_per_image)
                    pred = per_image_preds[0] if len(per_image_preds)==1 else per_image_preds[local_pos] 
                except Exception:
                    pred = per_image_preds[0]
                # Extract pred boxes tensor: attempt common attributes
                try:
                    boxes_tensor = pred.boxes.xyxy.cpu()  # Nx4
                    scores = pred.boxes.conf.cpu()
                    classes = pred.boxes.cls.cpu()
                    # filter by conf_thresh
                    keep = scores >= conf_thresh
                    boxes_tensor = boxes_tensor[keep]
                    classes = classes[keep]
                except Exception:
                    # fallback: try tensor output (N,6) x1,y1,x2,y2,conf,cls
                    try:
                        out = pred.cpu()
                        boxes_tensor = out[:, :4]
                        scores = out[:, 4]
                        keep = scores >= conf_thresh
                        boxes_tensor = boxes_tensor[keep]
                    except Exception:
                        boxes_tensor = torch.empty((0,4))
                # convert gt xywh -> xyxy (normalized)
                gt_xywh = gt_boxes
                gt_xyxy = torch.zeros_like(gt_xywh)
                # gt shape (N,4): x_center,y_center,w,h normalized
                gt_xyxy[:,0] = gt_xywh[:,0] - gt_xywh[:,2]/2
                gt_xyxy[:,1] = gt_xywh[:,1] - gt_xywh[:,3]/2
                gt_xyxy[:,2] = gt_xywh[:,0] + gt_xywh[:,2]/2
                gt_xyxy[:,3] = gt_xywh[:,1] + gt_xywh[:,3]/2
                # compute greedy matching
                matched_gt = set()
                tp = 0
                for pb in boxes_tensor:
                    if gt_xyxy.shape[0] == 0:
                        break
                    ious = iou_xyxy(pb, gt_xyxy)
                    best_i = torch.argmax(ious).item()
                    if ious[best_i] >= iou_thresh and best_i not in matched_gt:
                        tp += 1
                        matched_gt.add(best_i)
                fp = boxes_tensor.shape[0] - tp
                fn = gt_xyxy.shape[0] - len(matched_gt)
                P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                self.state_dict[img_id]['P'] = float(P)
                self.state_dict[img_id]['R'] = float(R)
        model.train()

class AFSSIndexSampler(Sampler):
    """
    Sampler that yields indices from active_indices. Use set_active_indices() to update per-epoch.
    This sampler is compatible with PyTorch BatchSampler used by DataLoader.
    """
    def __init__(self, num_samples, initial_indices=None, shuffle=True):
        self.num_samples = int(num_samples)
        if initial_indices is None:
            self.active_indices = list(range(self.num_samples))
        else:
            self.active_indices = list(initial_indices)
        self.shuffle = shuffle
    
    def set_active_indices(self, indices):
        self.active_indices = list(indices)

    def __iter__(self):
        idxs = self.active_indices.copy()
        if self.shuffle:
            random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.active_indices)





