"""
CNN .pth -> ONNX Exporter (hardcoded paths)
Just run: python cnn_to_onnx.py
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os

# ──────────────────────────────────────────────
# CONFIGURE THESE PATHS
# ──────────────────────────────────────────────
PTH_PATH    = r"B:\Licenta\Mega-Model\malware_cnn_best_v2.pth"
OUTPUT_PATH = r"B:\Licenta\3d.onnx"
INPUT_SIZE  = [3, 224, 224]   # [channels, height, width] - change if needed
NUM_CLASSES = 9                # detected from checkpoint classifier shape [9, 1792]
# ──────────────────────────────────────────────


def build_model(state_dict):
    # Detect EfficientNet variant by first conv output channels
    first_key = "base.features.0.0.weight"
    first_channels = state_dict[first_key].shape[0]

    variant_map = {
        48: "efficientnet_b4",
        40: "efficientnet_b3",
        32: "efficientnet_b0",
        24: "efficientnet_b1",
    }
    variant = variant_map.get(first_channels, "efficientnet_b4")
    print(f"[INFO] Detected variant: {variant} (first conv channels: {first_channels})")

    class WrappedEfficientNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = getattr(models, variant)(weights=None)
            in_features = self.base.classifier[1].in_features
            self.base.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

        def forward(self, x):
            return self.base(x)

    return WrappedEfficientNet()


def main():
    print(f"\n{'='*55}")
    print(f"  Loading: {PTH_PATH}")
    print(f"{'='*55}\n")

    if not os.path.exists(PTH_PATH):
        print(f"[ERROR] File not found: {PTH_PATH}")
        return

    # Load state dict
    state_dict = torch.load(PTH_PATH, map_location="cpu", weights_only=True)
    print(f"[INFO] State dict loaded — {len(state_dict)} keys")

    # Build model with correct NUM_CLASSES=9
    model = build_model(state_dict)

    # Load weights - use strict=False so size mismatches don't crash,
    # but since NUM_CLASSES=9 now matches the checkpoint it should be clean
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARNING] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[WARNING] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if not missing and not unexpected:
        print("[INFO] All weights loaded successfully — perfect match!")

    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total parameters: {total:,}")

    # Export to ONNX
    print(f"\n{'='*55}")
    print(f"  Exporting to ONNX -> {OUTPUT_PATH}")
    print(f"{'='*55}")

    dummy_input = torch.randn(1, *INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\n  ✅ Saved: {OUTPUT_PATH}  ({size_mb:.2f} MB)")
    print(f"  👉 Open it at https://netron.app to view your model!\n")

    # Validate
    try:
        import onnx
        onnx.checker.check_model(onnx.load(OUTPUT_PATH))
        print("  ✅ ONNX validation passed.\n")
    except ImportError:
        print("  [TIP] pip install onnx  to enable validation\n")
    except Exception as e:
        print(f"  [WARNING] Validation issue: {e}\n")


if __name__ == "__main__":
    main()