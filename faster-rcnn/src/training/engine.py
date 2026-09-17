"""Training and inference loops shared by ``train.py`` and ``evaluate.py``."""

import math
import time

import torch
from tqdm import tqdm


def move_targets(targets, device):
    return [
        {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
        for t in targets
    ]


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(step):
        if step >= warmup_iters:
            return 1.0
        alpha = step / max(warmup_iters, 1)
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    epochs,
    warmup_scheduler=None,
    clip_grad_norm=None,
    print_freq=50,
    max_skipped_frac=0.25,
):
    """One pass over the training set. Returns a dict of mean losses.

    Batches whose gradients come back non-finite are skipped rather than
    applied. That is a guard against the Apple-MPS backward bug documented in
    ``README.md``; on a healthy device it should never trigger, so the epoch
    aborts if more than ``max_skipped_frac`` of steps are skipped.
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

    for images, targets in progress:
        images = [img.to(device) for img in images]
        targets = move_targets(targets, device)

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        loss_value = float(losses.detach().cpu())

        if not math.isfinite(loss_value):
            raise RuntimeError(
                f"Loss became {loss_value} at step {n_batches + n_skipped}. The "
                "model weights are already corrupt, so training cannot continue. "
                "If you are on Apple MPS, rerun with --device cpu (see README)."
            )

        optimizer.zero_grad(set_to_none=True)
        losses.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_grad_norm if clip_grad_norm else float("inf")
        )

        if not torch.isfinite(grad_norm):
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)

            total_steps = n_batches + n_skipped
            if total_steps >= 20 and n_skipped / total_steps > max_skipped_frac:
                raise RuntimeError(
                    f"{n_skipped}/{total_steps} steps produced non-finite "
                    "gradients. This is the Apple-MPS backward bug, not a "
                    "learning-rate problem -- rerun with --device cpu."
                )

            continue

        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu())
        loss_sums["total"] = loss_sums.get("total", 0.0) + loss_value
        n_batches += 1

        if n_batches % print_freq == 0:
            progress.set_postfix(loss=f"{loss_sums['total'] / n_batches:.4f}")

    means = {k: v / max(n_batches, 1) for k, v in loss_sums.items()}
    means["epoch_seconds"] = time.time() - start
    means["skipped_steps"] = n_skipped

    return means


def gradient_backend_is_sane(model, batch, device, max_norm=1e4, tries=3):
    """Check that the device produces usable gradients before training starts.

    PyTorch's Apple-MPS backward pass currently returns non-finite or wildly
    inflated gradients for the Faster R-CNN ROI head (the RPN losses are fine,
    which is why the failure is silent). The forward pass and inference are
    unaffected. On a healthy device gradient norms for this dataset are single
    digits, so anything above ``max_norm`` means the backend is broken.

    Returns ``(is_sane, observed_norms)``. The model's gradients are cleared
    afterwards; its parameters are untouched.
    """
    model.train()

    images, targets = batch
    images = [img.to(device) for img in images]
    targets = move_targets(targets, device)

    norms = []

    for _ in range(tries):
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        model.zero_grad(set_to_none=True)
        losses.backward()

        norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        )
        norms.append(norm)

    model.zero_grad(set_to_none=True)

    is_sane = all(math.isfinite(n) and n < max_norm for n in norms)

    return is_sane, norms


@torch.inference_mode()
def predict_dataset(model, data_loader, device, score_thresh=0.0, desc="predict"):
    """Run inference over a loader.

    Returns ``(predictions, targets)``, both lists of CPU tensor dicts aligned
    with the loader's iteration order.
    """
    model.eval()

    predictions = []
    ground_truth = []

    for images, targets in tqdm(data_loader, desc=desc, leave=False, dynamic_ncols=True):
        images = [img.to(device) for img in images]

        outputs = model(images)

        for output, target in zip(outputs, targets):
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()

            if score_thresh > 0:
                keep = scores >= score_thresh
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            predictions.append(
                {"boxes": boxes, "scores": scores, "labels": labels}
            )
            ground_truth.append(
                {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                }
            )

    return predictions, ground_truth
