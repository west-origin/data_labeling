"""YOLOX 0.3.0 확장: 이미지에서 확인 안 된 클래스의 배경 손실을 빼는 데이터·손실 (ADR 0060).

train_yolox.py가 만든 Exp 파일이 이 모듈을 import한다 (torch·YOLOX가 있는 환경에서만 쓴다).

문제: Open Images·LVIS는 이미지마다 모든 클래스를 다 달지 않았다. YOLOX의 objectness는 클래스와
무관하게 "박스가 배정되지 않은 앵커 = 배경(0)"으로 학습하므로, 주석이 없는 우리 클래스 물체(예: LVIS
이미지 속 문, OI 이미지 속 밀대)가 배경으로 학습된다. YOLOX에는 이를 막는 기능이 없다.

여기서 하는 근사 (지어낸 정답이 아니라 손실을 빼는 것만 한다):
- 데이터: build_objects.py가 쓴 images[].verified_class_ids로 이미지마다 "확인 안 된 클래스"
  마스크를 만든다. 모자이크·mixup은 여러 이미지를 섞으므로 섞인 이미지들의 마스크
  합집합(하나라도 확인 안 됐으면 확인 안 됨)을 쓴다. 마스크는 라벨 텐서 뒤에 열로 붙여
  YOLOX 학습 루프를 그대로 통과시킨다.
- 손실: 박스가 배정되지 않은(배경) 앵커 가운데, 모델이 그 이미지에서 확인 안 된 클래스를
  `ignore_score`(objectness * 클래스 확률) 넘게 예측한 앵커는 objectness 배경 손실에서 뺀다.
  박스 배정(SimOTA)·박스·클래스 손실은 그대로다.
한계: 위치를 모르는 누락 물체를 모델 예측으로 추정하는 휴리스틱이다. 학습 초기(예측이 약할 때)에는
거의 작동하지 않고, 확인 안 된 클래스를 모델이 못 맞히면 여전히 배경으로 학습된다.
클래스 확률 음성 학습은 YOLOX에 원래 없으므로(클래스 손실은 배정된 앵커에만) 건드릴 것이 없다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from yolox.data import COCODataset
from yolox.models import YOLOXHead


class VerifiedCOCODataset(COCODataset):
    """COCODataset + 이미지별 확인 안 된 클래스 마스크, 그리고 pull_item 호출 기록."""

    def __init__(self, *args: Any, num_classes: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        index = {cid: i for i, cid in enumerate(self.class_ids)}
        self.unverified = np.ones((len(self.ids), num_classes), dtype=np.float32)
        for k, img_id in enumerate(self.ids):
            for cid in self.coco.imgs[img_id].get("verified_class_ids", []):
                self.unverified[k, index[cid]] = 0.0
        self.pulled: list[int] = []

    def pull_item(self, index: int) -> Any:
        # 모자이크·mixup이 이 함수로 원본 이미지를 꺼낸다 (DataLoader 작업자마다 한 번에 하나씩)
        self.pulled.append(index)
        return super().pull_item(index)


class UnverifiedMaskDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """MosaicDetection을 감싸 라벨 (max_labels, 5) 뒤에 마스크 (max_labels, C)를 붙인다."""

    def __init__(self, mosaic: Any, base: VerifiedCOCODataset) -> None:
        self.mosaic = mosaic
        self.base = base

    def __len__(self) -> int:
        return len(self.mosaic)

    def __getattr__(self, name: str) -> Any:
        if name in ("mosaic", "base"):
            raise AttributeError(name)
        return getattr(self.mosaic, name)

    def __getitem__(self, idx: Any) -> Any:
        self.base.pulled.clear()
        img, labels, info, img_id = self.mosaic[idx]
        used = sorted(set(self.base.pulled))
        if used:
            mask = self.base.unverified[used].max(axis=0)
        else:
            mask = np.zeros(self.base.unverified.shape[1], dtype=np.float32)
        tiled = np.broadcast_to(mask, (labels.shape[0], mask.shape[0]))
        labels = np.concatenate([labels, tiled.astype(labels.dtype)], axis=1)
        return img, labels, info, img_id


class MaskedBCE(nn.Module):
    """BCEWithLogits(reduction=none) + objectness 배경 손실 무시 마스크.

    YOLOX head는 같은 함수로 objectness(모양 (B*A, 1))와 클래스(모양 (fg, C))를 계산한다.
    마스크 모양 (B*A, 1)과 같은 입력에만 적용한다 (C > 1이라 겹치지 않는다).
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.obj_ignore: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce(inputs, target)
        ign = self.obj_ignore
        if ign is not None and loss.shape == ign.shape:
            loss = loss.masked_fill(ign & (target <= 0), 0.0)
        return loss


class FederatedYOLOXHead(YOLOXHead):
    """YOLOXHead + 확인 안 된 클래스 배경 objectness 손실 무시 (모듈 설명)."""

    def __init__(self, num_classes: int, *args: Any, ignore_score: float = 0.3, **kwargs: Any):
        if num_classes < 2:
            raise ValueError("FederatedYOLOXHead는 클래스가 2개 이상이어야 합니다")
        super().__init__(num_classes, *args, **kwargs)
        self.bcewithlog_loss = MaskedBCE()
        self.ignore_score = ignore_score
        self._unverified: torch.Tensor | None = None
        self.last_ignored_fraction = 0.0  # 마지막 배치에서 무시한 앵커 비율 (확인용)

    def forward(self, xin: Any, labels: torch.Tensor | None = None, imgs: Any = None) -> Any:
        if labels is not None and labels.shape[-1] > 5:
            self._unverified = labels[:, 0, 5:] > 0.5
            labels = labels[..., :5].contiguous()
        return super().forward(xin, labels, imgs)

    def get_losses(
        self,
        imgs: Any,
        x_shifts: Any,
        y_shifts: Any,
        expanded_strides: Any,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        origin_preds: Any,
        dtype: Any,
    ) -> Any:
        unv = self._unverified
        ign = None
        self.last_ignored_fraction = 0.0
        if unv is not None and bool(unv.any()) and self.ignore_score < 1.0:
            with torch.no_grad():
                # 학습 중 outputs[..., 4:]는 로짓이다 (박스만 디코드됨)
                score = outputs[:, :, 4:5].float().sigmoid() * outputs[:, :, 5:].float().sigmoid()
                score = score * unv[:, None, :].to(score.dtype)
                ign = (score.amax(dim=-1) > self.ignore_score).reshape(-1, 1)
            self.last_ignored_fraction = float(ign.float().mean())
        self.bcewithlog_loss.obj_ignore = ign
        try:
            return super().get_losses(
                imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype
            )
        finally:
            self.bcewithlog_loss.obj_ignore = None
            self._unverified = None


def build_model(exp: Any, federated: bool, ignore_score: float) -> nn.Module:
    """Exp.get_model과 같되 head만 바꾼다 (가중치 이름은 같아서 COCO 체크포인트를 그대로 읽는다)."""
    from yolox.models import YOLOPAFPN, YOLOX

    def init_yolo(m: nn.Module) -> None:
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eps = 1e-3
                mod.momentum = 0.03

    if getattr(exp, "model", None) is None:
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(exp.depth, exp.width, in_channels=in_channels, act=exp.act)
        if federated:
            head: nn.Module = FederatedYOLOXHead(
                exp.num_classes,
                exp.width,
                in_channels=in_channels,
                act=exp.act,
                ignore_score=ignore_score,
            )
        else:
            head = YOLOXHead(exp.num_classes, exp.width, in_channels=in_channels, act=exp.act)
        exp.model = YOLOX(backbone, head)
    exp.model.apply(init_yolo)
    exp.model.head.initialize_biases(1e-2)
    exp.model.train()
    return exp.model


def build_train_loader(
    exp: Any, batch_size: int, is_distributed: bool, no_aug: bool, cache_img: bool, federated: bool
) -> Any:
    """yolox_base.Exp.get_data_loader와 같되 데이터셋을 VerifiedCOCODataset(+마스크)으로 바꾼다."""
    import torch.distributed as dist
    from yolox.data import (
        DataLoader,
        InfiniteSampler,
        MosaicDetection,
        TrainTransform,
        YoloBatchSampler,
        worker_init_reset_seed,
    )
    from yolox.utils import wait_for_the_master

    with wait_for_the_master():
        base = VerifiedCOCODataset(
            data_dir=exp.data_dir,
            json_file=exp.train_ann,
            img_size=exp.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=exp.flip_prob, hsv_prob=exp.hsv_prob),
            cache=cache_img,
            num_classes=exp.num_classes,
        )
    mosaic = MosaicDetection(
        base,
        mosaic=not no_aug,
        img_size=exp.input_size,
        preproc=TrainTransform(max_labels=120, flip_prob=exp.flip_prob, hsv_prob=exp.hsv_prob),
        degrees=exp.degrees,
        translate=exp.translate,
        mosaic_scale=exp.mosaic_scale,
        mixup_scale=exp.mixup_scale,
        shear=exp.shear,
        enable_mixup=exp.enable_mixup,
        mosaic_prob=exp.mosaic_prob,
        mixup_prob=exp.mixup_prob,
    )
    dataset: Any = UnverifiedMaskDataset(mosaic, base) if federated else mosaic
    exp.dataset = dataset
    if is_distributed:
        batch_size = batch_size // dist.get_world_size()
    sampler = InfiniteSampler(len(dataset), seed=exp.seed if exp.seed else 0)
    batch_sampler = YoloBatchSampler(
        sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug
    )
    return DataLoader(
        dataset,
        num_workers=exp.data_num_workers,
        pin_memory=True,
        batch_sampler=batch_sampler,
        worker_init_fn=worker_init_reset_seed,
    )


def scale_targets(exp: Any, inputs: torch.Tensor, targets: torch.Tensor, tsize: Any) -> Any:
    """Exp.preprocess와 같되 박스 열(1~4)만 늘이고 줄인다 (뒤의 마스크 열은 그대로)."""
    scale_y = tsize[0] / exp.input_size[0]
    scale_x = tsize[1] / exp.input_size[1]
    if scale_x != 1 or scale_y != 1:
        inputs = nn.functional.interpolate(inputs, size=tsize, mode="bilinear", align_corners=False)
        targets[..., 1:5:2] = targets[..., 1:5:2] * scale_x
        targets[..., 2:5:2] = targets[..., 2:5:2] * scale_y
    return inputs, targets


def use_pycocotools_eval() -> None:
    """YOLOX의 C++ 빠른 COCOeval(실행 중 컴파일)을 끄고 pycocotools를 쓰게 한다.

    평가기는 `from yolox.layers import COCOeval_opt`가 ImportError면 pycocotools로 떨어진다.
    컴파일 실패로 학습 도중(첫 평가 때) 멈추는 일을 막는다.
    """
    import yolox.layers as layers

    if hasattr(layers, "COCOeval_opt"):
        delattr(layers, "COCOeval_opt")
