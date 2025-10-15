import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import eval_tta as eval_utils
from src.model import build_model
from src.dataset import SkinDataset
from src.transforms import AugConfig, build_valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temperature scaling calibration for ISIC model.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint to calibrate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for calibration pass.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--no-tta", action="store_true", help="Disable flip TTA during calibration.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    return parser.parse_args()


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracy = (predictions == labels)
    ece = 0.0
    total = len(labels)
    for idx in range(n_bins):
        mask = (confidences > bins[idx]) & (confidences <= bins[idx + 1])
        count = mask.sum()
        if count == 0:
            continue
        ece += abs(accuracy[mask].mean() - confidences[mask].mean()) * (count / total)
    return float(ece)


def find_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    temp = torch.nn.Parameter(torch.ones([]) * 1.0)
    optimizer = torch.optim.LBFGS([temp], lr=0.1, max_iter=50)

    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()

    def closure():
        optimizer.zero_grad()
        scaled = logits_t / temp.clamp_min(1e-3)
        loss = F.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temp.detach().item())


def main():
    args = parse_args()
    cfg = eval_utils.load_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = [str(c).strip().lower() for c in checkpoint.get("classes", cfg["classes"]["names"])]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    valid_df = eval_utils.prepare_validation(cfg, classes)
    data_cfg = cfg.get("data", {})
    img_size = int(data_cfg.get("img_size", 336))
    aug_cfg = AugConfig(
        img_size=img_size,
        mean=tuple(data_cfg.get("normalize_mean", (0.485, 0.456, 0.406))),
        std=tuple(data_cfg.get("normalize_std", (0.229, 0.224, 0.225))),
    )
    center_crop = bool(cfg.get("augmentation", {}).get("valid", {}).get("center_crop", True))
    transform = build_valid(aug_cfg, center_crop=center_crop)

    dataset = SkinDataset(
        valid_df,
        classes,
        img_size=img_size,
        transform=transform,
        hair_removal=False,
        images_dir=data_cfg.get("images_dir", ""),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=True if device.type == "cuda" else False,
    )

    model = build_model(
        cfg["train"]["model_name"],
        num_classes=len(classes),
        pretrained=False,
        compile_model=False,
    )
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    model.to(device)
    model.eval()

    use_tta = not args.no_tta
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device)
            logits = model(inputs)
            if use_tta:
                logits = 0.5 * (logits + model(torch.flip(inputs, dims=[3])))
            logits_all.append(logits.cpu().numpy())
            labels_all.append(targets.cpu().numpy())

    logits_all = np.concatenate(logits_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    pre_probs = torch.softmax(torch.from_numpy(logits_all), dim=1).numpy()
    pre_ece = expected_calibration_error(pre_probs, labels_all)

    temperature = find_temperature(logits_all, labels_all)
    post_probs = torch.softmax(torch.from_numpy(logits_all) / temperature, dim=1).numpy()
    post_ece = expected_calibration_error(post_probs, labels_all)

    output_path = args.output
    if output_path is None:
        deploy_cfg = cfg.get("deploy", {})
        default_path = deploy_cfg.get("temperature_file", "checkpoints/temperature.json")
        output_path = Path(default_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "temperature": temperature,
        "pre_ece": pre_ece,
        "post_ece": post_ece,
        "num_samples": int(labels_all.shape[0]),
        "use_tta": use_tta,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
