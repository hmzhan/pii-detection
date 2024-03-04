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
import config


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
        max_length=config.INFERENCE_MAX_LENGTH
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
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATHS.keys()[0])
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
    total_weight = sum(config.MODEL_PATHS.values())
    intermediate_dir = "./intermediate_predictions"
    os.makedirs(intermediate_dir, exist_ok=True)

    for idx, (model_path, weight) in enumerate(config.MODEL_PATHS.items()):
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
