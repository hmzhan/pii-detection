# Ensemble of four models for PII detection

import re
import os
import gc
import json
import argparse
import torch
import numpy as np
import pandas as pd
from itertools import chain
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from scipy.special import softmax
from ..config import *
from spacy.lang.en import English


nlp = English()


def tokenize(example, tokenizer):
    """
    Tokenize text in test data
    :param example: one example from test data
    :param tokenizer: tokenizer loaded from HuggingFace
    :return: tokenized text
    """
    text = []
    token_map = []
    idx = 0

    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):
        text.append(t)
        token_map.extend([idx] * len(t))
        if ws:
            text.append(" ")
            token_map.append(-1)
        idx += 1

    tokenized = tokenizer(
        "".join(text),
        return_offsets_mapping=True,
        trunation=False,
        max_length=INFERENCE_MAX_LENGTH
    )

    return {
        **tokenized,
        "token_map": token_map
    }


def tokenize_test_data(data):
    """
    Tokenize test data
    :param data: test data
    :return: tokenized data
    """
    dataset = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [x["document"] for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    })
    tokenizer = AutoTokenizer.from_pretrained(list(MODEL_PATHS.keys())[0])
    dataset = dataset.map(
        tokenize,
        fn_kwars={"tokenizer": tokenizer},
        num_proc=2
    )
    return dataset


def make_inference(dataset):
    """

    :param dataset:
    :return:
    """
    total_weight = sum(MODEL_PATHS.values())
    intermediate_dir = "./intermediate_predictions"
    os.makedirs(intermediate_dir, exist_ok=True)

    for idx, (model_path, weight) in enumerate(MODEL_PATHS.items()):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
        args = TrainingArguments(
            ".",
            per_device_eval_batch_size=1,
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=collator,
            tokenizer=tokenizer
        )
        predictions = trainer.predict(dataset).predictions
        weighted_predictions = softmax(predictions, axis=-1) * weight
        np.save(os.path.join(intermediate_dir, f'weighted_preds_{idx}.npy'), weighted_predictions)

        del model, trainer, tokenizer, predictions, weighted_predictions
        torch.cuda.empty_cache()
        gc.collect()

    aggregated_predictions = None
    for file_name in os.listdir(intermediate_dir):
        weighted_predictions = np.load(os.path.join(intermediate_dir, file_name))
        if aggregated_predictions is None:
            aggregated_predictions = weighted_predictions
        else:
            aggregated_predictions += weighted_predictions
    weighted_average_predictions = aggregated_predictions / total_weight

    return weighted_average_predictions


def get_final_prediction(weighted_average_predictions):
    """

    :return:
    """
    preds = weighted_average_predictions.argmax(-1)
    preds_without_o = weighted_average_predictions[:, :, :12].argmax(-1)
    o_preds = weighted_average_predictions[:, :, 12]
    threshold = 0.975
    return np.where(o_preds < threshold, preds_without_o, preds)


def process_final_predictions(model_path, preds_final, ds):
    """

    :param preds_final:
    :param ds:
    :return:
    """
    config = json.load(open(Path(model_path) / "config.json"))
    id2label = config["id2label"]

    pairs = set()
    processed = []
    for p, token_map, offsets, tokens, doc in zip(preds_final, ds["token_map"], ds["offset_mapping"],
                                                  ds["tokens"], ds["document"]):
        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[str(token_pred)]
            if start_idx + end_idx == 0:
                continue
            if token_map[start_idx] == -1:
                start_idx += 1
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1
            if start_idx >= len(token_map):
                break
            token_id = token_map[start_idx]

            if label_pred in ("O", "B-EMAIL", "B-PHONE_NUM", "I-PHONE_NUM") or token_id == -1:
                continue
            pair = (doc, token_id)
            if pair not in pairs:
                processed.append({
                    "document": doc,
                    "token": token_id,
                    "label": label_pred,
                    "token_str": tokens[token_id]
                })
                pairs.add(pair)
    return processed


def find_span(target: list[str], document: list[str]) -> list[list[int]]:
    """

    :param target:
    :param document:
    :return:
    """
    idx = 0
    spans = []
    span = []

    for i, token in enumerate(document):
        if token != target[idx]:
            idx = 0
            span = []
            continue
        span.append(i)
        idx += 1
        if idx == len(target):
            spans.append(span)
            span = []
            idx = 0
            continue
    return spans


def detect_email(data):
    """
    Detect email address in input documents
    :param data: test data
    :return:
    """
    email_regex = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    emails = []
    for _data in data:
        for token_idx, token in enumerate(_data["tokens"]):
            if re.fullmatch(email_regex, token) is not None:
                emails.append({
                    "document": _data["document"],
                    "token": token_idx,
                    "label": "B-EMAIL",
                    "token_str": token
                })
    return emails


def detect_phone_number(data):
    """
    Detect phone number in input documents
    :param data: test data
    :return:
    """
    phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")
    phone_nums = []
    for _data in data:
        matches = phone_num_regex.findall(_data["full_text"])
        if not matches:
            continue
        for match in matches:
            target = [t.text for t in nlp.tokenizer(match)]
            matched_spans = find_span(target, _data["tokens"])
        for matched_span in matched_spans:
            for intermediate, token_idx in enumerate(matched_span):
                prefix = "I" if intermediate else "B"
                phone_nums.append({
                    "document": _data["document"],
                    "token": token_idx,
                    "label": f"{prefix}-PHONE_NUM",
                    "token_str": _data["tokens"][token_idx]
                })
    return phone_nums



