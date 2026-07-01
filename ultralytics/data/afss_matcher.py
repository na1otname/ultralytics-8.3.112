import random
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torchvision.ops as tv_ops  # 新增：用于高效计算 IoU 矩阵
from torch.utils.data import Sampler
from ultralytics.utils import LOGGER, TQDM, ops
from ultralytics.utils.metrics import box_iou


def iou_xyxy(box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between a single box and multiple boxes. Boxes expected as xyxy.

    box1: (4,), boxes2: (N,4)
    """
    if boxes2.numel() == 0:
        return torch.tensor([], device=box1.device)
    lt = torch.max(box1[:2], boxes2[:, :2])
    rb = torch.min(box1[2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = a1 + a2 - inter
    return inter / union.clamp(min=1e-6)


def _unique_first(tensor: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask marking the first occurrence of each unique value.

    Args:
        tensor: 1-D GPU tensor.

    Returns:
        mask: boolean tensor, True for first occurrence.
    """
    device = tensor.device
    # Use torch.unique with stable=True to get indices of the first occurrence
    # in the original order (preserves GPU usage and avoids sorting-based bugs).
    try:
        _, first_idx = torch.unique(tensor, return_inverse=False, return_counts=False, return_index=True, stable=True)
    except TypeError:
        # Fallback for older PyTorch versions without `stable` argument: compute on CPU
        vals_cpu = tensor.cpu().numpy()
        seen = {}
        mask_list = [False] * len(vals_cpu)
        for i, v in enumerate(vals_cpu):
            if v not in seen:
                seen[v] = True
                mask_list[i] = True
        return torch.tensor(mask_list, dtype=torch.bool, device=device)

    result = torch.zeros(len(tensor), dtype=torch.bool, device=device)
    result[first_idx] = True
    return result


class AFSSManager:
    """Anti-Forgetting Sampling Strategy manager (P/R-based version).

    Key design choices:
    1. Dynamic sampling ratios: more data early, harder focus later.
    2. Single-threshold P/R (F1) for sufficiency — matches original AFSS paper.
       Using fixed-IoU Precision & Recall avoids the instability of per-image AP.
    3. Negative samples treated as moderate (not easy) to avoid over-suppression.
    4. Minimum per-epoch coverage guarantee so training never drops too low.
    5. Decoupled evaluate_and_update so caller can run it *before* get_epoch_subset.
    6. All computations run on GPU to avoid CPU-GPU transfer overhead.
    """

    def __init__(
        self,
        num_samples: int,
        easy_frac: float = 0.02,
        moderate_frac: float = 0.4,
        forced_mod_gap: int = 3,
        forced_easy_gap: int = 10,
    ):
        self.num_samples = int(num_samples)
        self.easy_frac = float(easy_frac)
        self.moderate_frac = float(moderate_frac)
        self.forced_mod_gap = int(forced_mod_gap)
        self.forced_easy_gap = int(forced_easy_gap)

        # state: sufficiency in [0,1], ep = last epoch this sample was trained
        self.state_dict = {i: {"P": 0.0,"R": 0.0, "ep": -1} for i in range(self.num_samples)}

        self.device = None

    def _prepare_batch(self, si, batch):
        """Prepare a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]

        device = bbox.device
        dtype = bbox.dtype

        imgsz = torch.tensor(batch["img"].shape[2:], device=device, dtype=dtype)
        ori_shape = torch.as_tensor(batch["ori_shape"][si], device=device, dtype=dtype)
        ratio_pad = torch.as_tensor(batch["ratio_pad"][si], device=device, dtype=dtype)

        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * imgsz[[1, 0, 1, 0]]
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)

        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepare predictions for evaluation against ground truth."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )
        return predn

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """Match predictions to ground truth objects using IoU at a single threshold (0.5).

        Returns a bool tensor of shape (n_dets,) marking which detections are correct.
        """
        device = pred_classes.device
        correct = torch.zeros(pred_classes.shape[0], dtype=torch.bool, device=device)
        correct_class = (true_classes[:, None] == pred_classes).to(device)
        iou = iou * correct_class

        if use_scipy:
            import scipy
            iou_np = iou.cpu().numpy()
            cost_matrix = iou_np * (iou_np >= 0.5)
            if cost_matrix.any():
                labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid]] = True
        else:
            matches = torch.nonzero(iou >= 0.5, as_tuple=False)
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    match_ious = iou[matches[:, 0], matches[:, 1]]
                    order = torch.argsort(match_ious, descending=True)
                    matches = matches[order]

                    mask1 = _unique_first(matches[:, 1])
                    matches = matches[mask1]
                    mask2 = _unique_first(matches[:, 0])
                    matches = matches[mask2]

                correct[matches[:, 1].long()] = True

        return correct

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """Return boolean vector marking which detections are correct (IoU>=0.5, class match)."""
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def get_epoch_subset(self, current_epoch: int) -> List[int]:
        """Return list of indices (omega) for training this epoch using AFSS."""
        
        omega = set()
        easy_pool, moderate_pool, hard_pool = [], [], []

        # ==========================================
        # 1. 学习充分度评估与分类 (Eq. 2)
        # ==========================================
        for img_id, state in self.state_dict.items():
            suff = min(state["P"], state["R"])
            if suff > 0.85:
                easy_pool.append(img_id)
            elif 0.55 <= suff <= 0.85:
                moderate_pool.append(img_id)
            else:
                hard_pool.append(img_id)

        # ==========================================
        # 核心策略 1: 困难样本全量参与 (Full Coverage - Eq. 16)
        # ==========================================
        omega.update(hard_pool)

        # ==========================================
        # 核心策略 2: 简单样本持续复习 (Continuous Review - Eq. 3, 4, 5, 6)
        # ==========================================
        forced_easy = [
            img for img in easy_pool
            if (current_epoch - 1) - self.state_dict[img]["ep"] >= self.forced_easy_gap
        ]
        omega.update(forced_easy)
        
        total_easy_target = max(0, int(self.easy_frac * len(easy_pool)) - len(forced_easy))
        
        # 从剩余简单样本中随机采样，补齐到 easy 目标数量
        remain_easy = list(set(easy_pool) - set(forced_easy))
        total_easy_target = min(total_easy_target, len(remain_easy))
        Ar = random.sample(remain_easy, total_easy_target) if total_easy_target > 0 else []
        omega.update(Ar)

        # ==========================================
        # 核心策略 3: 中等样本短期覆盖 (Short-Term Coverage - Eq. 7, 8, 9)
        # ==========================================
        forced_mod = [
            img for img in moderate_pool
            if (current_epoch - 1) - self.state_dict[img]["ep"] >= self.forced_mod_gap
        ]
        omega.update(forced_mod)

        M1 = int(self.moderate_frac * len(moderate_pool)) - len(forced_mod)
        M1 = max(0, M1)
        remain_mod = list(set(moderate_pool) - set(forced_mod))
        M1 = min(M1, len(remain_mod))
        Br = random.sample(remain_mod, M1) if M1 > 0 else []
        omega.update(Br)

        # ==========================================
        # 4. 状态更新 (State Update - Eq. 15)
        # ==========================================
        for img_id in omega:
            self.state_dict[img_id]["ep"] = current_epoch

        return list(omega)

    def print_sufficiency_distribution(self, current_epoch: Optional[int] = None):
        easy, mod, hard = 0, 0, 0
        for s in self.state_dict.values():
            suff = min(s["P"], s["R"])
            if suff > 0.85:
                easy += 1
            elif 0.55 <= suff <= 0.85:
                mod += 1
            else:
                hard += 1
        total = self.num_samples
        if total == 0:
            LOGGER.info("  AFSS State Dict is empty!")
            return
        LOGGER.info(f"  -> Easy     (>0.85): {easy:5d} / {total} ({easy/total*100:.1f}%)")
        LOGGER.info(f"  -> Moderate (0.55-0.85): {mod:5d} / {total} ({mod/total*100:.1f}%)")
        LOGGER.info(f"  -> Hard     (<0.55): {hard:5d} / {total} ({hard/total*100:.1f}%)")

    @torch.no_grad()
    def evaluate_and_update(self, model, dataloader, conf_thresh=0.2, iou_thresh=0.5):
        """Run inference over dataloader and update sufficiency for each image index."""

        model_eval = model.module if hasattr(model, "module") else model
        pbar = TQDM(dataloader, total=len(dataloader), desc="AFSS Evaluating")
        model.eval()

        self.device = next(model_eval.parameters()).device

        dataset = getattr(dataloader, "dataset", None)

        file_to_idx = {}
        stem_to_idx = {}
        if dataset is not None and hasattr(dataset, "im_files"):
            for idx, f in enumerate(dataset.im_files):
                f_str = str(f)
                file_to_idx[f_str] = idx
                stem_to_idx[Path(f_str).stem] = idx

        for _, batch in enumerate(pbar):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device=self.device, non_blocking=True)
            imgs = batch["img"].to(dtype=next(model.parameters()).dtype) / 255.0

            preds = model_eval(imgs)
            per_image_preds = ops.non_max_suppression(preds, conf_thresh, iou_thresh)

            for i, pred in enumerate(per_image_preds):
                npr = len(pred)

                pbatch = self._prepare_batch(i, batch)
                cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
                predn = self._prepare_pred(pred, pbatch)

                ngt = int(len(cls))

                # =====================================================================
                # 🚀 性能优化：通过 Dict.get() 代替原来的 List.index() 与双重循环
                # =====================================================================
                img_id = None
                if "im_file" in batch:
                    im_file_s = str(batch["im_file"][i])
                    # 1. 尝试精确路径查找
                    img_id = file_to_idx.get(im_file_s)
                    # 2. 备用模糊 Stem 查找
                    if img_id is None:
                        img_id = stem_to_idx.get(Path(im_file_s).stem)
                # =====================================================================

                if img_id is None or img_id not in self.state_dict:
                    continue

                # 场景 1：无 Ground Truth (背景图)
                if ngt == 0:
                    if npr == 0:
                        # 既没物体也没预测：完美
                        self.state_dict[img_id]["P"] = 1.0
                        self.state_dict[img_id]["R"] = 1.0
                    else:
                        # 没物体却预测了：全是误报 (FP)
                        self.state_dict[img_id]["P"] = 0.0
                        self.state_dict[img_id]["R"] = 0.0
                    continue

                # 场景 2：有 Ground Truth 但模型一个框都没预测出来 (全漏检)
                if npr == 0:
                    self.state_dict[img_id]["P"] = 0.0
                    self.state_dict[img_id]["R"] = 0.0
                    continue
                    
                # =====================================================================
                # 场景 3：严谨的基于类别和 IoU 的一对一匹配 (修正了 P、R 的计算)
                # =====================================================================
                TP = 0
                FP = 0
                FN = 0

                pred_classes = predn[:, 5]
                gt_classes = cls.view(-1)
                
                # 获取该图片中出现的所有类别 (预测类并集真实类)
                unique_classes = torch.unique(torch.cat([pred_classes, gt_classes]))

                for c in unique_classes:
                    mask_p = (pred_classes == c)
                    mask_g = (gt_classes == c)
                    
                    p_boxes = predn[mask_p, :4]
                    g_boxes = bbox[mask_g]
                    
                    n_p = len(p_boxes)
                    n_g = len(g_boxes)
                    
                    if n_p == 0:
                        # 预测无该类，但 GT 有 -> 全是漏检
                        FN += n_g
                        continue
                        
                    if n_g == 0:
                        # 预测有该类，但 GT 无 -> 全是误报
                        FP += n_p
                        continue
                        
                    # 计算 IoU 矩阵
                    ious = tv_ops.box_iou(p_boxes, g_boxes)
                    
                    matched_gts = set()
                    
                    # 贪心匹配 (按预测框的顺序，NMS结果默认是按置信度降序的)
                    for p_idx in range(n_p):
                        max_iou, max_g_idx = ious[p_idx].max(dim=0)
                        max_g_idx_val = max_g_idx.item()
                        
                        if max_iou >= iou_thresh and max_g_idx_val not in matched_gts:
                            matched_gts.add(max_g_idx_val)
                            TP += 1
                        else:
                            FP += 1
                            
                    # GT中未被匹配掉的数量，即为漏检
                    FN += (n_g - len(matched_gts))

                P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                R = TP / (TP + FN) if (TP + FN) > 0 else 0.0

                self.state_dict[img_id]["P"] = P
                self.state_dict[img_id]["R"] = R

        model.train()


class AFSSIndexSampler(Sampler):
    """Simple sampler for single-process training: yields only active indices (shuffled)."""

    def __init__(self, num_samples: int, initial_indices: Optional[List[int]] = None, shuffle: bool = True):
        super().__init__(None)
        self.num_samples = int(num_samples)
        self.active_indices = list(range(self.num_samples)) if initial_indices is None else list(initial_indices)
        self.shuffle = shuffle

    def set_active_indices(self, indices: List[int]):
        self.active_indices = list(indices)

    def __iter__(self):
        idxs = self.active_indices.copy()
        if self.shuffle:
            random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.active_indices)


class AFSSDistributedSampler(Sampler):
    """Sampler compatible with Distributed training that samples only from active_indices and shards per rank."""

    def __init__(
        self,
        dataset,
        active_indices: Optional[List[int]] = None,
        rank: Optional[int] = None,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle
        self.rank = int(rank) if rank is not None else 0
        self.world_size = int(world_size)
        self.seed = int(seed or 0)
        self.epoch = 0
        self.active_indices = list(active_indices) if active_indices is not None else list(range(len(dataset)))
        self.local_indices = []
        self._rebuild_local_indices()

    def _rebuild_local_indices(self):
        inds = list(self.active_indices)
        if self.shuffle:
            rnd = random.Random(self.seed + self.epoch)
            rnd.shuffle(inds)
        self.local_indices = inds[self.rank::self.world_size]

    def set_active_indices(self, active_indices):
        if active_indices is None:
            return
        self.active_indices = list(active_indices)
        self._rebuild_local_indices()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._rebuild_local_indices()

    def __iter__(self):
        for idx in self.local_indices:
            yield idx

    def __len__(self):
        return len(self.local_indices)