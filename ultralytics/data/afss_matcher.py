import random
import torch
import warnings
import numpy as np
from torch.utils.data import Sampler
from typing import List, Optional
from ultralytics.utils import LOGGER, TQDM, ops
from ultralytics.utils.metrics import box_iou
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
        # IoU thresholds used for matching (mAP-style 10 thresholds by default)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.device = None

    
    def _prepare_batch(self, si, batch):
        """
        Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (dict): Batch data containing images and annotations.

        Returns:
            (dict): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}
    
    def _prepare_pred(self, pred, pbatch):
        """
        Prepare predictions for evaluation against ground truth.

        Args:
            pred (torch.Tensor): Model predictions.
            pbatch (dict): Prepared batch information.

        Returns:
            (torch.Tensor): Prepared predictions in native space.
        """
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn
    
    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """
        Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)
            
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
    def evaluate_and_update(self, model, dataloader, conf_thresh=0.2, iou_thresh=0.5):
        """Run inference over dataloader and update P,R for each image index.

        Notes:
            - dataloader should yield batches with keys: 'img', 'bboxes', 'batch_idx'
            - GT format is expected to be normalized xywh in `batch['bboxes']`
            - model(...) should return ultralytics-like Results per image (or a list/iterable)
        """
        pbar = TQDM(dataloader, total=len(dataloader), desc="AFSS Evaluating")
        model.eval()

        # set device and ensure iou vector exists
        self.device = next(model.parameters()).device
        if not hasattr(self, "iouv") or self.iouv is None:
            self.iouv = torch.linspace(0.5, 0.95, 10)

        # dataset reference for mapping image file -> global index
        dataset = getattr(dataloader, "dataset", None)

        seen = 0
        for si, batch in enumerate(pbar):
            imgs = batch["img"].to(device=self.device, dtype=next(model.parameters()).dtype) / 255.0

            # model output: normalize to per-image iterable
            preds = model(imgs)
            per_image_preds = ops.non_max_suppression(preds, conf_thresh, iou_thresh)

            for i, pred in enumerate(per_image_preds):
                seen += 1
                npr = len(pred)

                pbatch = self._prepare_batch(i, batch)
                cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
                bbox = bbox.to(self.device)
                cls = cls.to(self.device)
                predn = self._prepare_pred(pred, pbatch)

                # skip images without ground truth (no meaningful recall)
                ngt = int(len(cls))
                if ngt == 0:
                    continue

                # compute true positives matrix for all IoU thresholds
                if npr == 0:
                    tp_count = 0
                else:
                    correct = self._process_batch(predn, bbox, cls)  # (npr, n_iou)
                    # pick the IoU threshold nearest to iou_thresh
                    thr_idx = int((self.iouv.cpu() - float(iou_thresh)).abs().argmin().item())
                    tp_count = int(correct[:, thr_idx].sum().item())

                fp = max(0, npr - tp_count)
                fn = max(0, ngt - tp_count)

                P = tp_count / (tp_count + fp) if (tp_count + fp) > 0 else 0.0
                R = tp_count / (tp_count + fn) if (tp_count + fn) > 0 else 0.0

                # map batch image -> global image id using image file list when available
                img_id = None
                if "im_file" in batch:
                    try:
                        im_file = batch["im_file"][i]
                        # normalize to str for matching
                        im_file_s = str(im_file)
                        if dataset is not None and hasattr(dataset, "im_files"):
                            try:
                                img_id = int(dataset.im_files.index(im_file_s))
                            except ValueError:
                                # sometimes im_file may already be a Path or differ in formatting
                                # try matching by stem (filename without suffix)
                                from pathlib import Path

                                stem = Path(im_file_s).stem
                                for idx, f in enumerate(dataset.im_files):
                                    if Path(f).stem == stem:
                                        img_id = idx
                                        break
                    except Exception:
                        img_id = None

                # update AFSS state for this image if mapped
                if img_id is not None and img_id in self.state_dict:
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
