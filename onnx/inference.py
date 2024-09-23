import os
import argparse
import typing as T

import cv2
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def draw_boxes_to_image(
    image: np.ndarray,
    boxes: np.ndarray,
    phrases: T.List[str],
    confs: T.List[float],
) -> None:
    for box, phrase, conf in zip(boxes, phrases, confs):
        box = box.astype(np.int32)

        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        image = cv2.putText(
            image, f"{phrase} ({conf:.2f})", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )


def get_phrases_from_posmap(
    posmap: np.ndarray,
    tokenized: T.Dict,
    tokenizer: AutoTokenizer,
    left_idx: int = 0,
    right_idx: int = 255,
) -> str:
    assert isinstance(posmap, np.ndarray), "posmap must be numpy array"
    assert posmap.ndim == 1, "posmap must be 1-dim"

    posmap[0 : left_idx + 1] = False
    posmap[right_idx:] = False
    non_zero_idx = np.nonzero(posmap)[0]
    token_ids = tokenized["input_ids"][0, non_zero_idx]
    return tokenizer.decode(token_ids)


def generate_masks_with_special_tokens_and_transfer_map(
    tokenized: T.Dict[str, np.ndarray], special_tokens_list: T.List[int]
):
    """Generate attention mask between each pair of special tokens"""
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    assert bs == 1, "Batch size must be 1"
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = np.zeros((1, num_token), dtype=bool)  # [bs, num_token]
    for special_token in special_tokens_list:
        special_tokens_mask = np.logical_or(special_tokens_mask, input_ids == special_token)

    # get indexes of special tokens
    rows, cols = np.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = np.expand_dims(np.eye(num_token, dtype=bool), axis=0)  # [bs, num_token, num_token]
    position_ids = np.zeros((1, num_token), dtype=np.int64)  # [bs, num_token]
    previous_col = 0
    for row, col in zip(rows, cols):
        if col == 0 or col == num_token - 1:
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = np.arange(0, col - previous_col)
        previous_col = col
    return attention_mask, position_ids


def infer(args) -> None:
    # Load model
    sess = ort.InferenceSession(args.model_path)

    # Load image
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = 640, 800
    image = cv2.resize(image, (w, h))

    # Preprocess image
    x = image / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0).astype(np.float32)

    # Prompt text
    caption = args.text_prompt
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    # Preprocess text
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)
    tokenized = tokenizer(caption, padding="longest", return_tensors="np")
    specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

    max_text_len = 256
    if tokenized["input_ids"].shape[1] > max_text_len:
        for k in tokenized:
            tokenized[k] = tokenized[k][:, :max_text_len]
    tokenized["attention_mask"] = tokenized["attention_mask"].astype(bool)

    (text_token_mask, position_ids) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens
    )

    # Run model
    output = sess.run(
        None,
        {
            "image": x,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "position_ids": position_ids,
            "token_type_ids": tokenized["token_type_ids"],
            "text_token_mask": text_token_mask,
        },
    )

    proba = output[0][0]  # (nq, 256)
    boxes = output[1][0]  # (nq, 4)

    # filter output
    mask = proba.max(axis=1) > args.box_threshold
    proba = proba[mask]
    boxes = boxes[mask]

    # get phrase
    phrases, confs = [], []
    for i, prob in enumerate(proba):
        confs.append(prob.max())
        phrase = get_phrases_from_posmap(prob > args.text_threshold, tokenized, tokenizer)
        phrases.append(phrase)
        # from 0..1 to 0..W, 0..H
        boxes[i] = boxes[i] * [w, h, w, h]
        # from xywh to xyxy
        boxes[i][:2] -= boxes[i][2:] / 2
        boxes[i][2:] += boxes[i][:2]

    # Draw boxes
    draw_boxes_to_image(image, boxes, phrases, confs)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, "output.jpg"), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--model_path", "-p", type=str, required=True, help="path to onnx file")
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    args = parser.parse_args()

    infer(args)
