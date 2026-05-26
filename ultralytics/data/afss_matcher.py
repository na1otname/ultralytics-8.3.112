import random
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
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
    sorted_vals, sort_idx = torch.sort(tensor)
    mask = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        sorted_vals[1:] != sorted_vals[:-1]
    ])
    result = torch.zeros(len(tensor), dtype=torch.bool, device=device)
    result[sort_idx[mask]] = True
    return result


class AFSSManager:
    """Anti-Forgetting Sampling Strategy manager (improved version).

    Key improvements over v1:
    1. Dynamic sampling ratios: more data early, harder focus later.
    2. Multi-threshold AP for sufficiency instead of single-threshold P/R.
    3. Negative samples treated as moderate (not easy) to avoid over-suppression.
    4. Minimum per-epoch coverage guarantee so training never drops too low.
    5. Decoupled evaluate_and_update so caller can run it *before* get_epoch_subset.
    6. All precision computations run on GPU to avoid CPU-GPU transfer overhead.
    """

    def __init__(
        self,
        num_samples: int,
        easy_frac: float = 0.15,
        moderate_frac: float = 0.7,
        forced_mod_gap: int = 3,
        forced_easy_gap: int = 10,
        min_coverage: float = 0.60,
        total_epochs: Optional[int] = None,
    ):
        self.num_samples = int(num_samples)
        self.easy_frac = float(easy_frac)
        self.moderate_frac = float(moderate_frac)
        self.forced_mod_gap = int(forced_mod_gap)
        self.forced_easy_gap = int(forced_easy_gap)
        self.min_coverage = float(min_coverage)
        self.total_epochs = int(total_epochs) if total_epochs else None

        # state: sufficiency in [0,1], ep = last epoch this sample was trained
        self.state_dict = {i: {"suff": 0.0, "ep": -1} for i in range(self.num_samples)}

        # IoU thresholds (mAP-style 10 thresholds)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.device = None

    def _get_dynamic_fracs(self, progress_ratio: float):
        """Return (easy_frac, moderate_frac) scaled by training progress.

        Early epochs: keep more data so the model learns a stable baseline.
        Late epochs: gradually reduce to focus on hard cases.
        """
        p = float(np.clip(progress_ratio, 0.0, 1.0))
        # easy:  start at full easy_frac, decay to 30% of it by end
        e = self.easy_frac * (1.0 - 0.70 * p)
        # moderate: start at full moderate_frac, decay to 50% of it by end
        m = self.moderate_frac * (1.0 - 0.50 * p)
        return e, m

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
        """Match predictions to ground truth objects using IoU. Entirely on GPU."""
        n_dets = pred_classes.shape[0]
        n_iou = self.iouv.shape[0]
        device = pred_classes.device

        correct = torch.zeros((n_dets, n_iou), dtype=torch.bool, device=device)
        correct_class = (true_classes[:, None] == pred_classes).to(device)
        iou = iou * correct_class

        for i, threshold in enumerate(self.iouv.tolist()):
            if use_scipy:
                import scipy
                iou_np = iou.cpu().numpy()
                cost_matrix = iou_np * (iou_np >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = torch.nonzero(iou >= threshold, as_tuple=False)
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        match_ious = iou[matches[:, 0], matches[:, 1]]
                        order = torch.argsort(match_ious, descending=True)
                        matches = matches[order]

                        mask1 = _unique_first(matches[:, 1])
                        matches = matches[mask1]
                        mask2 = _unique_first(matches[:, 0])
                        matches = matches[mask2]

                    correct[matches[:, 1].long(), i] = True

        return correct

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """Return correct prediction matrix."""
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    @staticmethod
    def _compute_ap_torch(recall: torch.Tensor, precision: torch.Tensor) -> float:
        """Compute AP using 101-point interpolation, entirely on GPU."""
        device = recall.device
        mrec = torch.cat([torch.zeros(1, device=device), recall, torch.ones(1, device=device)])
        mpre = torch.cat([torch.ones(1, device=device), precision, torch.zeros(1, device=device)])

        mpre = torch.flip(torch.cummax(torch.flip(mpre, dims=[0]), dim=0).values, dims=[0])

        x = torch.linspace(0, 1, 101, device=device)
        indices = torch.searchsorted(mrec, x, right=True) - 1
        indices = torch.clamp(indices, 0, len(mpre) - 1)
        y = mpre[indices]

        dx = x[1] - x[0]
        ap = (y[1:] + y[:-1]).sum() * 0.5 * dx
        return ap.item()

    def get_epoch_subset(self, current_epoch: int) -> List[int]:
        """Return list of indices (omega) for training this epoch."""
        progress = current_epoch / self.total_epochs if self.total_epochs else 0.0
        easy_frac_dyn, moderate_frac_dyn = self._get_dynamic_fracs(progress)

        omega = set()
        easy_pool, moderate_pool, hard_pool = [], [], []
        for img_id, state in self.state_dict.items():
            suff = state["suff"]
            if suff > 0.85:
                easy_pool.append(img_id)
            elif 0.55 <= suff <= 0.85:
                moderate_pool.append(img_id)
            else:
                hard_pool.append(img_id)

        # 1. Include all hard
        omega.update(hard_pool)

        # 2. Forced moderate
        forced_mod = [img for img in moderate_pool if current_epoch - 1 - self.state_dict[img]["ep"] >= self.forced_mod_gap]
        omega.update(forced_mod)

        # 3. Supplement moderate
        remain_mod = list(set(moderate_pool) - set(forced_mod))
        M1 = max(0, int(moderate_frac_dyn * len(moderate_pool)) - len(forced_mod))
        if M1 > 0 and remain_mod:
            omega.update(random.sample(remain_mod, min(M1, len(remain_mod))))

        # 4. Forced easy
        forced_easy = [img for img in easy_pool if current_epoch - 1 - self.state_dict[img]["ep"] >= self.forced_easy_gap]
        max_forced_easy = int(0.5 * easy_frac_dyn * len(easy_pool)) if len(easy_pool) > 0 else 0
        if len(forced_easy) > max_forced_easy and max_forced_easy > 0:
            forced_easy = random.sample(forced_easy, max_forced_easy)
        omega.update(forced_easy)

        # 5. Supplement easy
        remain_easy = list(set(easy_pool) - set(forced_easy))
        E2 = max(0, int(easy_frac_dyn * len(easy_pool)) - len(forced_easy))
        if E2 > 0 and remain_easy:
            omega.update(random.sample(remain_easy, min(E2, len(remain_easy))))

        # 6. Minimum coverage guarantee
        min_size = max(1, int(self.min_coverage * self.num_samples))
        if len(omega) < min_size:
            not_selected = [i for i in range(self.num_samples) if i not in omega]
            need = min_size - len(omega)
            if not_selected and need > 0:
                omega.update(random.sample(not_selected, min(need, len(not_selected))))

        # 7. Update epoch record
        for img_id in omega:
            self.state_dict[img_id]["ep"] = current_epoch

        return list(omega)

    def print_sufficiency_distribution(self, current_epoch: Optional[int] = None):
        easy, mod, hard = 0, 0, 0
        for s in self.state_dict.values():
            suff = s["suff"]
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
        if current_epoch is not None:
            omega = self.get_epoch_subset(current_epoch)
            LOGGER.info(f"  -> Selected this epoch: {len(omega):5d} / {total} ({len(omega)/total*100:.1f}%)")

    @torch.no_grad()
    def evaluate_and_update(self, model, dataloader, conf_thresh=0.2, iou_thresh=0.5):
        """Run inference over dataloader and update sufficiency for each image index."""
        pbar = TQDM(dataloader, total=len(dataloader), desc="AFSS Evaluating")
        model.eval()

        self.device = next(model.parameters()).device
        if not hasattr(self, "iouv") or self.iouv is None:
            self.iouv = torch.linspace(0.5, 0.95, 10)
        self.iouv = self.iouv.to(self.device)

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

            preds = model(imgs)
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

                if ngt == 0:
                    if npr == 0:
                        self.state_dict[img_id]["suff"] = 0.55
                    else:
                        self.state_dict[img_id]["suff"] = 0.0
                    continue

                if npr == 0:
                    self.state_dict[img_id]["suff"] = 0.0
                    continue

                correct = self._process_batch(predn, bbox, cls)
                conf = predn[:, 4]

                ap_per_iou = []
                for j in range(correct.shape[1]):
                    tpc = correct[:, j].float()
                    if tpc.sum() == 0:
                        ap_per_iou.append(0.0)
                        continue

                    order = torch.argsort(conf, descending=True)
                    tpc_sorted = tpc[order]

                    tpc_cumsum = torch.cumsum(tpc_sorted, dim=0)
                    fpc_cumsum = torch.cumsum(1.0 - tpc_sorted, dim=0)

                    recall = tpc_cumsum / ngt
                    precision = tpc_cumsum / (tpc_cumsum + fpc_cumsum + 1e-16)

                    ap = self._compute_ap_torch(recall, precision)
                    ap_per_iou.append(ap)

                sufficiency = float(np.mean(ap_per_iou))
                self.state_dict[img_id]["suff"] = sufficiency

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