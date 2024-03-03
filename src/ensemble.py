# Ensemble of four models for PII detection

import re
import os
import json
import argparse
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

