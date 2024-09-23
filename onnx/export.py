import os
import argparse

import torch

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict


class Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        proba = outputs["pred_logits"].sigmoid()
        boxes = outputs["pred_boxes"]
        return proba, boxes


def load_model(model_config_path: str, model_checkpoint_path: str, cpu_only: bool = False) -> torch.nn.Module:
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"

    args.use_checkpoint = False
    args.use_transformer_ckpt = False

    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return Wrapper(model)


def export(model: torch.nn.Module, output_dir: str) -> None:
    caption = "the running dog ."
    input_ids = model.model.tokenizer([caption], return_tensors="pt")["input_ids"]
    position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]])
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[True, True, True, True, True, True]])
    text_token_mask = torch.tensor(
        [
            [
                [True, False, False, False, False, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, False, False, False, False, True],
            ]
        ]
    )

    image = torch.randn(1, 3, 640, 800)
    dynamic_axes = {
        "input_ids": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        "position_ids": {1: "seq_len"},
        "token_type_ids": {1: "seq_len"},
        "text_token_mask": {1: "seq_len", 2: "seq_len"},
    }

    torch.onnx.export(
        model,
        f=f"{output_dir}/dino.onnx",
        args=(
            image,
            input_ids,
            attention_mask,
            position_ids,
            token_type_ids,
            text_token_mask,
        ),
        input_names=[
            "image",
            "input_ids",
            "attention_mask",
            "position_ids",
            "token_type_ids",
            "text_token_mask",
        ],
        output_names=["proba", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=16,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export Grounding DINO Model to IR", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    args = parser.parse_args()
    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    model = load_model(config_file, checkpoint_path, cpu_only=True)
    export(model, output_dir)
