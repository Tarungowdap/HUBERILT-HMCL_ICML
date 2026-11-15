"""Training and evaluation for HMCL, Dynamic-Curvature, Euclidean, and EWC baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from projector.memory import MemoryProjector
from trainer.ewc import EWC
from trainer.head import EuclideanHead, HMCLHead
from utils import lorentz as L
from utils.config import ClassificationConfig, RetrievalConfig
from utils.drift_metrics import compute_representation_drift


class ContinualTrainer:
    """Train adaptation heads with geometric covariance projection, dynamic curvature, or baselines."""

    def __init__(
        self,
        classification_cfg: ClassificationConfig,
        retrieval_cfg: RetrievalConfig,
        variance_ratio: float = 0.9,
        pullback_beta: float = 0.1,
        method: str = "hmcl",
        curvature_min: float = 0.05,
        curvature_max: float = 0.5,
        curvature_lr: float = 1e-4,
        curvature_reg: float = 1.0,
        ewc_lambda: float = 100.0,
        ewc_gamma: float = 1.0,
    ) -> None:
        self.classification_cfg = classification_cfg
        self.retrieval_cfg = retrieval_cfg
        self.variance_ratio = float(variance_ratio)
        self.pullback_beta = float(pullback_beta)
        self.method = method.lower()
        self.curvature_min = float(curvature_min)
        self.curvature_max = float(curvature_max)
        self.curvature_lr = float(curvature_lr)
        self.curvature_reg = float(curvature_reg)
        self.ewc_lambda = float(ewc_lambda)
        self.ewc_gamma = float(ewc_gamma)

        self.device = self._resolve_device(classification_cfg.device)
        self.projector: MemoryProjector | None = None
        self.ewc: EWC | None = EWC(ewc_lambda=ewc_lambda, ewc_gamma=ewc_gamma) if self.method == "ewc" else None
        self.task_id = 0
        self.curvature_history: list[float] = []
        self.initial_task_representations: dict[int, torch.Tensor] = {}

    @staticmethod
    def _snapshot_head_matrix(model: nn.Module) -> torch.Tensor:
        """Freeze the task-start head matrix used as the pullback anchor."""
        if hasattr(model, "head") and hasattr(model.head, "linear"):
            return model.head.linear.weight.detach().clone()
        elif hasattr(model, "linear"):
            return model.linear.weight.detach().clone()
        return torch.empty(0)

    @staticmethod
    @torch.no_grad()
    def _apply_head_matrix_pullback(model: nn.Module, anchor: torch.Tensor, beta: float) -> None:
        """Interpolate the head matrix toward its task-start value."""
        if beta <= 0.0 or anchor.numel() == 0:
            return
        param = model.head.linear.weight if hasattr(model, "head") else model.linear.weight
        param.lerp_(anchor.to(device=param.device, dtype=param.dtype), beta)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def build_class_text(text_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Average the training text features belonging to each class."""
        num_classes = int(labels.max().item() + 1)
        class_features = text_features.new_zeros((num_classes, text_features.shape[-1]))
        for class_id in range(num_classes):
            indices = (labels == class_id).nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                raise ValueError(f"No text features found for class {class_id}")
            class_features[class_id] = text_features.index_select(0, indices).mean(dim=0)
        return class_features

    @staticmethod
    def _align_retrieval_text(train: dict[str, object], text_features: torch.Tensor) -> torch.Tensor:
        """Select the first stored caption associated with each training image."""
        if "gt_pairs" in train:
            indices = torch.tensor([p[1] for p in train["gt_pairs"]], dtype=torch.long, device=text_features.device)
            return text_features.index_select(0, indices)

        image_ids = [int(v) for v in train.get("image_ids", range(text_features.shape[0]))]
        text_ids = [int(v) for v in train.get("text_ids", range(text_features.shape[0]))]
        ground_truth = train.get("image_to_text_gt", {})
        text_positions = {t_id: pos for pos, t_id in enumerate(text_ids)}

        selected_positions = []
        for img_id in image_ids:
            candidates = [
                text_positions[t_id]
                for t_id in ground_truth.get(img_id, ())
                if t_id in text_positions
            ]
            if not candidates:
                selected_positions.append(0)
            else:
                selected_positions.append(min(candidates))

        indices = torch.tensor(selected_positions, dtype=torch.long, device=text_features.device)
        return text_features.index_select(0, indices)

    @staticmethod
    def _compute_recall(predictions: dict[int, list[int]], ground_truth: dict[int, set[int]], k: int) -> float:
        correct = sum(
            bool(set(predictions.get(query_id, [])[:k]) & paired_ids)
            for query_id, paired_ids in ground_truth.items()
        )
        return 100.0 * correct / max(1, len(ground_truth))

    def _contrastive_epoch(
        self,
        model: nn.Module,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        cfg: ClassificationConfig | RetrievalConfig,
    ) -> float:
        """Accumulate symmetric contrastive gradient over mini-batches."""
        indices = torch.randperm(image_features.shape[0], device=image_features.device)
        total_loss = 0.0
        num_batches = 0
        is_euclidean = isinstance(model, EuclideanHead)

        for start in range(0, indices.numel(), cfg.batch_size):
            batch_indices = indices[start : start + cfg.batch_size]
            image_batch = image_features.index_select(0, batch_indices)
            text_batch = text_features.index_select(0, batch_indices)

            image_z = model.encode_image(image_batch)
            text_z = model.encode_text(text_batch)

            if is_euclidean:
                logits = (image_z @ text_z.T) * model.logit_scale.exp()
                curv_val = 0.0
            else:
                curv_val = model.curv
                logits = -L.pairwise_dist(image_z, text_z, curv_val) * model.logit_scale.exp()

            targets = torch.arange(logits.shape[0], device=logits.device)
            loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

            if not is_euclidean and cfg.entail_weight > 0:
                loss = loss + cfg.entail_weight * L.compute_entailment_loss(text_z, image_z, curv_val)

            if self.method == "dynamic-curvature" and hasattr(model, "curvature_module"):
                anchor_loss = model.curvature_module.get_anchor_loss(self.curvature_reg)
                loss = loss + anchor_loss

            if self.method == "ewc" and self.ewc is not None and self.ewc.is_active():
                loss = loss + self.ewc.penalty(model)

            loss.backward()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    @staticmethod
    def _projector_input(features: torch.Tensor, embed_dim: int, curv: float) -> torch.Tensor:
        features = features.detach().cpu()
        return L.concat_time_dimension(features, curv)

    def fit_task(
        self,
        train: dict[str, object],
        model: nn.Module | None = None,
        task_type: str = "classification",
    ) -> nn.Module:
        """Fit one task, then update memory projector or EWC Fisher information."""
        cfg = self.classification_cfg if task_type == "classification" else self.retrieval_cfg
        feats = train["features"] if "features" in train else train["image_features"]
        image_features = feats.to(self.device)
        labels = train["labels"].long() if "labels" in train else None

        if "text_features" in train:
            stored_text_features = train["text_features"]
        elif labels is not None:
            stored_text_features = self.build_class_text(image_features.cpu(), labels)[labels]
        else:
            stored_text_features = image_features.cpu()

        text_features = stored_text_features
        if task_type == "retrieval":
            text_features = self._align_retrieval_text(train, text_features)
        text_features = text_features.to(self.device)
        embed_dim = image_features.shape[-1]

        if model is None:
            if self.method == "euclidean":
                model = EuclideanHead(embed_dim=embed_dim).to(self.device)
            else:
                is_dynamic = self.method == "dynamic-curvature"
                model = HMCLHead(
                    embed_dim=embed_dim,
                    curv=cfg.curv,
                    dynamic_curvature=is_dynamic,
                    curvature_min=self.curvature_min,
                    curvature_max=self.curvature_max,
                ).to(self.device)
            model.log_architecture()
            if hasattr(model, "curv"):
                self.curvature_history.append(float(model.curv.item()))

        if hasattr(model, "set_task_start"):
            model.set_task_start()

        params = []
        if hasattr(model, "head") and hasattr(model.head, "linear"):
            params.append({"params": [model.head.linear.weight], "lr": cfg.lr, "weight_decay": cfg.weight_decay})
        elif hasattr(model, "linear"):
            params.append({"params": [model.linear.weight], "lr": cfg.lr, "weight_decay": cfg.weight_decay})

        if hasattr(model, "logit_scale"):
            params.append({"params": [model.logit_scale], "lr": cfg.lr, "weight_decay": 0.0})

        if self.method == "dynamic-curvature" and hasattr(model, "curvature_module") and model.curvature_module.learnable:
            params.append({"params": [model.curvature_module.raw_c], "lr": self.curvature_lr, "weight_decay": 0.0})

        optimizer = torch.optim.AdamW(params)

        use_projector = self.method in {"hmcl", "dynamic-curvature"}
        shared_projector = None
        if use_projector:
            if self.projector is None:
                self.projector = MemoryProjector(embed_dim, self.device)
            if self.task_id > 0:
                cov = self.projector.get_covariance()
                shared_projector = self.projector.calculate_projector(cov, self.variance_ratio)

        pullback_anchor = self._snapshot_head_matrix(model)

        for epoch in range(cfg.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            avg_loss = self._contrastive_epoch(model, image_features, text_features, cfg)

            if use_projector and shared_projector is not None:
                if hasattr(model, "head") and model.head.linear.weight.grad is not None:
                    grad = model.head.linear.weight.grad
                    grad.copy_(self.projector.project_gradient(grad, shared_projector))

            optimizer.step()

            if use_projector:
                self._apply_head_matrix_pullback(model, pullback_anchor, self.pullback_beta)

            logger.info(
                "Task {} ({}) epoch {}/{}: loss={:.4f}",
                self.task_id + 1,
                self.method,
                epoch + 1,
                cfg.epochs,
                avg_loss,
            )

        if use_projector and self.projector is not None:
            with torch.no_grad():
                curv_val = float(model.curv.item()) if hasattr(model, "curv") else 0.1
                img_aug = self._projector_input(image_features, embed_dim, curv_val)
                txt_aug = self._projector_input(stored_text_features, embed_dim, curv_val)
                self.projector.update_covariance(img_aug, txt_aug)

        elif self.method == "ewc" and self.ewc is not None:
            from torch.utils.data import DataLoader, TensorDataset
            ds = TensorDataset(image_features.cpu(), text_features.cpu())
            loader = DataLoader(ds, batch_size=cfg.batch_size)

            def crit(m, b):
                bx, by = b[0].to(self.device), b[1].to(self.device)
                z_x, z_y = m.encode_image(bx), m.encode_text(by)
                logits = -L.pairwise_dist(z_x, z_y, cfg.curv) * m.logit_scale.exp()
                t = torch.arange(logits.shape[0], device=logits.device)
                return 0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t))

            self.ewc.compute_fisher(model, loader, crit, self.device)

        if hasattr(model, "curv"):
            c_now = float(model.curv.item())
            self.curvature_history.append(c_now)
            logger.info("Task {} complete. Learned curvature: {:.4f}", self.task_id + 1, c_now)

        self.task_id += 1
        return model

    @torch.inference_mode()
    def evaluate_classification(
        self,
        model: nn.Module,
        split: dict[str, torch.Tensor],
        class_text: torch.Tensor,
    ) -> float:
        model.eval()
        feats = split["features"] if "features" in split else split["image_features"]
        image_z = model.encode_image(feats.to(self.device))
        text_z = model.encode_text(class_text.to(self.device))
        if isinstance(model, EuclideanHead):
            logits = image_z @ text_z.T
        else:
            curv_val = model.curv if hasattr(model, "curv") else self.classification_cfg.curv
            logits = -L.pairwise_dist(image_z, text_z, curv_val)
        predictions = logits.argmax(dim=-1)
        labels = split["labels"].long().to(self.device)
        return 100.0 * (predictions == labels).float().mean().item()

    @torch.inference_mode()
    def evaluate_retrieval(
        self,
        model: nn.Module,
        split: dict[str, object],
        ks: tuple[int, ...] = (1, 5, 10),
    ) -> dict[str, float]:
        model.eval()
        img_feats = split["features"] if "features" in split else split["image_features"]
        txt_feats = split["text_features"]
        image_z = model.encode_image(img_feats.to(self.device))
        text_z = model.encode_text(txt_feats.to(self.device))
        is_euclidean = isinstance(model, EuclideanHead)
        curv = float(model.curv.item()) if hasattr(model, "curv") else float(split.get("curv", self.retrieval_cfg.curv))

        if "gt_pairs" in split:
            image_ids = torch.arange(image_z.shape[0])
            text_ids = torch.arange(text_z.shape[0])
            i2t_gt = {i: {j} for i, j in split["gt_pairs"]}
            t2i_gt = {j: {i} for i, j in split["gt_pairs"]}
        else:
            image_ids = torch.as_tensor(split["image_ids"])
            text_ids = torch.as_tensor(split["text_ids"])
            i2t_gt = split["image_to_text_gt"]
            t2i_gt = split["text_to_image_gt"]

        text_to_image: dict[int, list[int]] = {}
        for query_ids, queries in zip(text_ids.split(256), text_z.split(256)):
            if is_euclidean:
                scores = queries @ image_z.T
            else:
                scores = -L.pairwise_dist(queries, image_z, curv)
            order = scores.argsort(dim=1, descending=True).cpu()
            retrieved_ids = image_ids[order]
            for query_id, result in zip(query_ids, retrieved_ids):
                text_to_image[int(query_id)] = result.tolist()

        image_to_text: dict[int, list[int]] = {}
        for query_ids, queries in zip(image_ids.split(256), image_z.split(256)):
            if is_euclidean:
                scores = queries @ text_z.T
            else:
                scores = -L.pairwise_dist(queries, text_z, curv)
            order = scores.argsort(dim=1, descending=True).cpu()
            retrieved_ids = text_ids[order]
            for query_id, result in zip(query_ids, retrieved_ids):
                image_to_text[int(query_id)] = result.tolist()

        results: dict[str, float] = {}
        for k in ks:
            results[f"t2i_r{k}"] = self._compute_recall(text_to_image, t2i_gt, k)
            results[f"i2t_r{k}"] = self._compute_recall(image_to_text, i2t_gt, k)
        return results
