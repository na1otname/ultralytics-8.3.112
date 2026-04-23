import random
import torch
import warnings
import numpy as np
from torch.utils.data import Sampler
from typing import List, Optional
from ultralytics.utils import LOGGER
from torch import distributed as dist


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


class AFSSManager:
    """Anti-Forgetting Sampling Strategy manager.

    Tracks per-image P/R and last-epoch seen; produces per-epoch active index subset.
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
        # initialize state: P,R in [0,1], ep last epoch updated
        self.state_dict = {i: {"P": 0.0, "R": 0.0, "ep": -1} for i in range(self.num_samples)}

    def get_epoch_subset(self, current_epoch: int) -> List[int]:
        """Return list of indices (omega) for training this epoch and update ep for chosen indices."""
        omega = set()
        easy_pool, moderate_pool, hard_pool = [], [], []
        for img_id, state in self.state_dict.items():
            sufficiency = min(state["P"], state["R"])
            if sufficiency > 0.85:
                easy_pool.append(img_id)
            elif 0.55 <= sufficiency <= 0.85:
                moderate_pool.append(img_id)
            else:
                hard_pool.append(img_id)

        # 1. Include all hard
        omega.update(hard_pool)

        # 2. Forced moderate (forgotten > forced_mod_gap)
        forced_mod = [img for img in moderate_pool if current_epoch - 1 - self.state_dict[img]["ep"] >= self.forced_mod_gap]
        omega.update(forced_mod)

        # 3. Supplement moderate up to moderate_frac
        remain_mod = list(set(moderate_pool) - set(forced_mod))
        M1 = max(0, int(self.moderate_frac * len(moderate_pool)) - len(forced_mod))
        if M1 > 0 and remain_mod:
            omega.update(random.sample(remain_mod, min(M1, len(remain_mod))))

        # 4. Forced easy (forgotten > forced_easy_gap), capped
        forced_easy = [img for img in easy_pool if current_epoch - 1 - self.state_dict[img]["ep"] >= self.forced_easy_gap]
        max_forced_easy = int(0.5 * self.easy_frac * len(easy_pool)) if len(easy_pool) > 0 else 0
        if len(forced_easy) > max_forced_easy and max_forced_easy > 0:
            forced_easy = random.sample(forced_easy, max_forced_easy)
        omega.update(forced_easy)

        # 5. Supplement easy to reach easy_frac
        remain_easy = list(set(easy_pool) - set(forced_easy))
        E2 = max(0, int(self.easy_frac * len(easy_pool)) - len(forced_easy))
        if E2 > 0 and remain_easy:
            omega.update(random.sample(remain_easy, min(E2, len(remain_easy))))

        # 6. Update epoch record
        for img_id in omega:
            self.state_dict[img_id]["ep"] = current_epoch

        return list(omega)

    def print_sufficiency_distribution(self):
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
    def evaluate_and_update(self, model, dataloader, conf_thresh=0.2, iou_thresh=0.5, device=None):
        """Run inference over dataloader and update P,R for each image index.

        Notes:
            - dataloader should yield batches with keys: 'img', 'bboxes', 'batch_idx'
            - GT format is expected to be normalized xywh in `batch['bboxes']`
            - model(...) should return ultralytics-like Results per image (or a list/iterable)
        """
        model.eval()
        dev = device if device is not None else getattr(model, "device", None)
        if dev is None:
            dev = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
        
        for batch in dataloader:
            imgs = batch['img'].to(device=next(model.parameters()).device, 
                        dtype=next(model.parameters()).dtype) / 255.0
            batch_idx = batch["batch_idx"]
            # model output: try to normalize to per-image iterable
            preds = model(imgs)
            per_image_preds = preds if isinstance(preds, (list, tuple)) else [preds]

            unique_ids = torch.unique(batch_idx)
            for img_id in unique_ids.tolist():
                mask = batch_idx == img_id
                gt_boxes = batch.get("bboxes", None)
                if gt_boxes is None or gt_boxes[mask].numel() == 0:
                    # no GT: mark as easy to avoid forcing
                    self.state_dict[img_id]["P"] = 1.0
                    self.state_dict[img_id]["R"] = 1.0
                    continue

                gt_boxes_img = gt_boxes[mask].cpu()
                # try pick corresponding prediction: assume per_image_preds order matches batch images
                # fallback to first prediction if mismatch
                try:
                    # approximate index by the first occurrence position in batch
                    first_pos = (batch_idx == img_id).nonzero(as_tuple=False)[0].item()
                    pred = per_image_preds[0] if len(per_image_preds) == 1 else per_image_preds[first_pos]
                except Exception:
                    pred = per_image_preds[0]

                # extract predicted boxes
                img_mask = (batch_idx == img_id).cpu()
                try:
                    all_boxes = pred.boxes.xyxy.cpu()
                    all_scores = pred.boxes.conf.cpu()
                    boxes_tensor = all_boxes[img_mask]
                    scores = all_scores[img_mask]
                except Exception:
                    try:
                        out = pred.cpu()
                        if out.ndim == 2:
                            # [total_pred, 6]
                            all_boxes = out[:, :4]
                            all_scores = out[:, 4]
                            boxes_tensor = all_boxes[img_mask]
                            scores = all_scores[img_mask]
                        elif out.ndim == 3 and out.shape[0] == imgs.shape[0]:
                            # [batch, num_pred, 6]
                            single = out[first_pos]
                            boxes_tensor = single[:, :4]
                            scores = single[:, 4]
                        else:
                            # try [batch, 4, num_pred]
                            if out.shape[1] == 4:
                                single = out[first_pos]  # [4, num_pred]
                                boxes_tensor = single.T  # [num_pred, 4]
                                scores = torch.ones(boxes_tensor.shape[0])  # dummy scores
                            else:
                                boxes_tensor = torch.zeros((0, 4))
                                scores = torch.zeros((0,))
                    except Exception:
                        boxes_tensor = torch.zeros((0, 4))
                        scores = torch.zeros((0,))

                keep = scores >= conf_thresh if scores.numel() else torch.tensor([], dtype=torch.bool)
                if keep.numel():
                    boxes_tensor = boxes_tensor[keep]
                else:
                    boxes_tensor = boxes_tensor

                # convert gt xywh -> xyxy
                gt_xywh = gt_boxes_img
                gt_xyxy = torch.zeros_like(gt_xywh)
                gt_xyxy[:, 0] = gt_xywh[:, 0] - gt_xywh[:, 2] / 2
                gt_xyxy[:, 1] = gt_xywh[:, 1] - gt_xywh[:, 3] / 2
                gt_xyxy[:, 2] = gt_xywh[:, 0] + gt_xywh[:, 2] / 2
                gt_xyxy[:, 3] = gt_xywh[:, 1] + gt_xywh[:, 3] / 2

                matched_gt = set()
                tp = 0
                for pb in boxes_tensor:
                    if gt_xyxy.shape[0] == 0:
                        break
                    ious = iou_xyxy(pb, gt_xyxy)
                    if ious.numel() == 0:
                        continue
                    best_i = torch.argmax(ious).item()
                    if ious[best_i] >= iou_thresh and best_i not in matched_gt:
                        tp += 1
                        matched_gt.add(best_i)
                fp = max(0, boxes_tensor.shape[0] - tp)
                fn = max(0, gt_xyxy.shape[0] - len(matched_gt))
                P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                self.state_dict[img_id]["P"] = float(P)
                self.state_dict[img_id]["R"] = float(R)

        model.train()


class AFSSIndexSampler(Sampler):
    """Simple sampler for single-process training: yields only active indices (shuffled)."""

    def __init__(self, num_samples: int, initial_indices: Optional[List[int]] = None, shuffle: bool = True):
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
    """Sampler compatible with Distributed training that samples only from active_indices and shards per rank.

    Behavior mirrors torch.utils.data.distributed.DistributedSampler but operates on a dynamic active_indices list.
    """

    def __init__(
        self,
        dataset,
        active_indices: Optional[List[int]] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.active_indices = list(range(len(dataset))) if active_indices is None else list(active_indices)

    def set_active_indices(self, indices: List[int]):
        self.active_indices = list(indices)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        # copy active indices
        indices = list(self.active_indices)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(indices), generator=g).tolist() if len(indices) > 0 else []
            indices = [indices[i] for i in perm]

        # add extra samples to make it evenly divisible
        if not self.drop_last:
            rem = (-len(indices)) % self.num_replicas
            if rem:
                # pad with indices from the beginning
                indices += indices[:rem]
        else:
            # drop tail so that evenly divisible
            indices = indices[: (len(indices) // self.num_replicas) * self.num_replicas]

        # subsample
        indices = indices[self.rank:: self.num_replicas]
        return iter(indices)

    def __len__(self):
        if self.drop_last:
            return len(self.active_indices) // self.num_replicas
        return (len(self.active_indices) + self.num_replicas - 1) // self.num_replicas
