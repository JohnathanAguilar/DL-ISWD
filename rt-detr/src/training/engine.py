"""Training and inference loops for the HuggingFace RT-DETR model."""

import math
import time

import torch
from tqdm import tqdm


def build_param_groups(model, lr, backbone_lr_mult, weight_decay):
    """Split parameters into backbone / rest, matching RT-DETR's official recipe
    of training the (pretrained) CNN backbone at a lower LR than the randomly
    initialized encoder/decoder.
    """
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    return [
        {"params": backbone_params, "lr": lr * backbone_lr_mult},
        {"params": other_params, "lr": lr},
    ]


def gradient_backend_is_sane(model, batch, device, max_norm=1e7):
    """Pre-flight check that ``device`` produces usable gradients.

    Mirrors ``faster-rcnn/engine.py``'s check. RT-DETR is expected to pass on
    MPS once the float64 position-embedding patch in ``model.py`` is applied
    (see that module's docstring) -- this just guards against a future
    transformers/torch version reintroducing a similar gap.

    ``max_norm`` is set high on purpose: RT-DETR's un-clipped gradient norms
    are legitimately large before the optimizer's own ``clip_grad_norm``
    (5e2-1e5 was observed on a healthy device during benchmarking here), so
    this only needs to catch actual corruption -- which showed up as literal
    ``nan`` or ~1e8-1e9 in the analogous Faster R-CNN/MPS bug -- not normal
    transformer-scale gradients.
    """
    model.train()

    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch.get("pixel_mask")
    pixel_mask = pixel_mask.to(device) if pixel_mask is not None else None
    labels = [
        {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in label.items()}
        for label in batch["labels"]
    ]

    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

    model.zero_grad(set_to_none=True)
    outputs.loss.backward()

    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
    model.zero_grad(set_to_none=True)

    is_sane = math.isfinite(grad_norm) and grad_norm < max_norm

    return is_sane, grad_norm


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    epochs,
    warmup_scheduler=None,
    clip_grad_norm=0.1,
    print_freq=50,
    max_skipped_frac=0.25,
):
    """One pass over the training set. Returns a dict of mean losses.

    Batches with non-finite gradients are skipped (see
    ``faster-rcnn/engine.py`` for the rationale); the epoch aborts if too
    many are skipped, since on a healthy setup this should not happen.
    """
    model.train()

    loss_sums = {}
    n_batches = 0
    n_skipped = 0
    start = time.time()

    progress = tqdm(
        data_loader,
        desc=f"epoch {epoch + 1}/{epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        pixel_mask = pixel_mask.to(device) if pixel_mask is not None else None
        labels = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in label.items()}
            for label in batch["labels"]
        ]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss

        loss_value = float(loss.detach().cpu())

        if not math.isfinite(loss_value):
            raise RuntimeError(
                f"Loss became {loss_value} at step {n_batches + n_skipped}. The "
                "model weights are already corrupt, so training cannot continue."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_grad_norm if clip_grad_norm else float("inf")
        )

        if not torch.isfinite(grad_norm):
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)

            total_steps = n_batches + n_skipped
            if total_steps >= 20 and n_skipped / total_steps > max_skipped_frac:
                raise RuntimeError(
                    f"{n_skipped}/{total_steps} steps produced non-finite gradients."
                )

            continue

        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        for k, v in outputs.loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu())
        loss_sums["total"] = loss_sums.get("total", 0.0) + loss_value
        n_batches += 1

        if n_batches % print_freq == 0:
            progress.set_postfix(loss=f"{loss_sums['total'] / n_batches:.4f}")

    means = {k: v / max(n_batches, 1) for k, v in loss_sums.items()}
    means["epoch_seconds"] = time.time() - start
    means["skipped_steps"] = n_skipped

    return means


@torch.inference_mode()
def predict_dataset(model, dataset, data_loader, processor, device, conf_thresh=0.001, desc="predict"):
    """Run inference over a loader.

    Returns ``(predictions, targets)``, both lists of CPU tensor dicts aligned
    with ``dataset`` order (recovered via each sample's ``image_id``, so this
    is safe even if the loader shuffles).
    """
    model.eval()

    n = len(dataset)
    predictions = [None] * n
    targets = [None] * n

    for batch in tqdm(data_loader, desc=desc, leave=False, dynamic_ncols=True):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        pixel_mask = pixel_mask.to(device) if pixel_mask is not None else None
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = [tuple(int(v) for v in label["orig_size"].tolist()) for label in labels]

        results = processor.post_process_object_detection(
            outputs, threshold=conf_thresh, target_sizes=target_sizes, use_focal_loss=True
        )

        for label, result in zip(labels, results):
            idx = int(label["image_id"])

            predictions[idx] = {
                "boxes": result["boxes"].detach().cpu(),
                "scores": result["scores"].detach().cpu(),
                "labels": result["labels"].detach().cpu(),
            }

            gt_boxes = dataset.ground_truth_boxes(idx)
            targets[idx] = {
                "boxes": gt_boxes,
                "labels": torch.zeros((len(gt_boxes),), dtype=torch.int64),
            }

    return predictions, targets
