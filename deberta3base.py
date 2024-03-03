# File comments


import json
import numpy as np
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification
)
from datasets import (
    Dataset,
    features
)
from seqeval.metrics import (
    recall_score,
    precision_score,
    classification_report,
    f1_score
)
from functools import partial
from itertools import chain

TARGET = [
    'B-EMAIL',
    'B-ID_NUM',
    'B-NAME_STUDENT',
    'B-PHONE_NUM',
    'B-STREET_ADDRESS',
    'B-URL_PERSONAL',
    'B-USERNAME',
    'I-ID_NUM',
    'I-NAME_STUDENT',
    'I-PHONE_NUM',
    'I-STREET_ADDRESS',
    'I-URL_PERSONAL'
]

ALL_LABELS = [
    'B-EMAIL',
    'B-ID_NUM',
    'B-NAME_STUDENT',
    'B-PHONE_NUM',
    'B-STREET_ADDRESS',
    'B-URL_PERSONAL',
    'B-USERNAME',
    'I-ID_NUM',
    'I-NAME_STUDENT',
    'I-PHONE_NUM',
    'I-STREET_ADDRESS',
    'I-URL_PERSONAL',
    'O'
]

LABEL2ID = {
    'B-EMAIL': 0,
    'B-ID_NUM': 1,
    'B-NAME_STUDENT': 2,
    'B-PHONE_NUM': 3,
    'B-STREET_ADDRESS': 4,
    'B-URL_PERSONAL': 5,
    'B-USERNAME': 6,
    'I-ID_NUM': 7,
    'I-NAME_STUDENT': 8,
    'I-PHONE_NUM': 9,
    'I-STREET_ADDRESS': 10,
    'I-URL_PERSONAL': 11,
    'O': 12
}

ID2LABEL = {
    0: 'B-EMAIL',
    1: 'B-ID_NUM',
    2: 'B-NAME_STUDENT',
    3: 'B-PHONE_NUM',
    4: 'B-STREET_ADDRESS',
    5: 'B-URL_PERSONAL',
    6: 'B-USERNAME',
    7: 'I-ID_NUM',
    8: 'I-NAME_STUDENT',
    9: 'I-PHONE_NUM',
    10: 'I-STREET_ADDRESS',
    11: 'I-URL_PERSONAL',
    12: 'O'
}


def compute_metrics(p, all_labels):
    """

    :param p:
    :param all_labels:
    :return:
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens), WHY???
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5 * 5) * recall * precision / (5 * 5 * precision + recall)

    return {
        'recall': recall,
        'precision': precision,
        'f1': f1_score
    }


class Data:
    # Class comments
    def __init__(self):
        self.TRAINING_MODEL_PATH = "microsoft/deberta-v3-base"
        self.TRAINING_MAX_LENGTH = 1024
        self.OUTPUT_DIR = "output"
        self.original_data_path = "/kaggle/input/pii-detection-removal-from-educational-data/train.json"
        self.external_data_path = "/kaggle/input/fix-punctuation-tokenization-external-dataset/pii_dataset_fixed.json"
        self.more_data_path = "/kaggle/input/fix-punctuation-tokenization-external-dataset/moredata_dataset_fixed.json"
        self.tokenizer = AutoTokenizer.from_pretrained(self.TRAINING_MODEL_PATH)
        self.raw_data = self.load_data(self.original_data_path, self.external_data_path, self.more_data_path)
        self.all_labels = sorted(list(set(chain(*[x["labels"] for x in self.raw_data]))))
        self.label2id = {l: i for i, l in enumerate(self.all_labels)}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.tokenized_data = self.tokenize_input_text(self.raw_data)

    @staticmethod
    def load_data(original_data_path, external_data_path, more_data_path):
        """
        Load training data
        :param original_data_path:
        :param external_data_path:
        :param more_data_path:
        :return:
        """
        original_data = json.load(open(original_data_path))
        pos = []
        neg = []
        for d in original_data:
            if any(np.array(d['labels']) != 'O'):
                pos.append(d)
            else:
                neg.append(d)
        external_data = json.load(
            open(external_data_path))
        more_data = json.load(open(more_data_path))

        return pos + neg[:len(neg) // 3] + external_data + more_data

    @staticmethod
    def _tokenize(example, tokenizer, label2id, max_length):
        """
        Tokenize original text for model training

        :param example: An example from input data: a dictionary
        :param tokenizer: tokenizer loaded from Huggingface: Deberta-v3-base
        :param label2id: label to id
        :param max_length: max length for tokenizer
        :return:
        """

        # Rebuild text from tokens
        texts = []
        labels = []
        for t, l, ws in zip(example['tokens'], example['provided_labels'], example['trailing_whitespace']):
            texts.append(t)
            labels.extend([l] * len(t))  # why add label for each character?
            if ws:
                texts.append(" ")
                labels.append("O")
        labels = np.array(labels)

        # Actual tokenization: tokenize the new text
        tokenized_text = tokenizer("".join(texts), return_offsets_mapping=True, max_length=max_length)
        texts = "".join(texts)
        token_labels = []

        for start_idx, end_idx in tokenized_text.offset_mapping:
            # cls token
            if start_idx == 0 and end_idx == 0:
                token_labels.append(label2id["O"])
                continue
            # case when token starts with whitespace
            if texts[start_idx].isspace():
                start_idx += 1
            token_labels.append(label2id[labels[start_idx]])

        length = len(tokenized_text.input_ids)
        return {**tokenized_text, "labels": token_labels, "length": length}

    def tokenize_input_text(self, data):
        """

        :param data:
        :return:
        """
        ds = Dataset.from_dict({
            "full_text": [x["full_text"] for x in data],
            "document": [str(x["document"]) for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
            "provided_labels": [x["labels"] for x in data]
        })
        return ds.map(
            self._tokenize,
            fn_kwargs={"tokenizer": self.tokenizer, "label2id": LABEL2ID, "max_length": self.TRAINING_MAX_LENGTH},
            num_proc=3
        )


class Model:
    def __init__(self):
        self.TRAINING_MODEL_PATH = "microsoft/deberta-v3-base"
        self.TRAINING_MAX_LENGTH = 1024
        self.OUTPUT_DIR = "output"
        self.tokenizer = AutoTokenizer.from_pretrained(self.TRAINING_MODEL_PATH)
        self.model = self.load_model()
        self.collator = self.create_collator()

    def load_model(self):
        return AutoModelForTokenClassification.from_pretrained(
            self.TRAINING_MODEL_PATH,
            num_labels=len(ALL_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True
        )

    def create_collator(self):
        return DataCollatorForTokenClassification(
            self.tokenizer,
            pad_to_multiple_of=16
        )


class Deberta3base:
    def __init__(self):
        self.TRAINING_MODEL_PATH = "microsoft/deberta-v3-base"
        self.TRAINING_MAX_LENGTH = 1024
        self.OUTPUT_DIR = "output"
        self.tokenizer = AutoTokenizer.from_pretrained(self.TRAINING_MODEL_PATH)

    def fit_model(self, model, data):
        args = TrainingArguments(
            output_dir=self.OUTPUT_DIR,
            fp16=True,
            learning_rate=2e-5,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            report_to="none",
            evaluation_strategy="no",
            do_eval=False,
            save_total_limit=1,
            logging_steps=20,
            lr_scheduler_type='cosine',
            metric_for_best_model="f1",
            greater_is_better=True,
            warmup_ratio=0.1,
            weight_decay=0.01
        )
        trainer = Trainer(
            model=model.model,
            args=args,
            train_dataset=data.tokenized_data,
            data_collator=model.collator,
            tokenizer=self.tokenizer,
            compute_metrics=partial(compute_metrics, all_labels=ALL_LABELS)
        )
        trainer.train()
        trainer.save_model("deberta3base_1024")
        self.tokenizer.save_pretrained("deberta3base_1024")

