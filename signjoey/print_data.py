# coding: utf-8
"""
Data module
"""
import os
import sys
import random
import torch
from torchtext import data
from torchtext.data import Dataset, Iterator
from dataset import SignTranslationDataset
from vocabulary import build_vocab, Vocabulary, UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN

def load_data(data_cfg: dict, num_samples: int = 1) -> (Dataset, Dataset, Dataset, Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Additionally, print a portion of loaded data for inspection.
    
    :param data_cfg: configuration dictionary for data ("data" part of configuration file)
    :param num_samples: number of samples to display for inspection
    :return: 
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
        - gls_vocab: gloss vocabulary extracted from training data
        - txt_vocab: spoken text vocabulary extracted from training data
    """
    data_path = data_cfg.get("data_path", "./data")

    train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]] if isinstance(data_cfg["train"], list) else os.path.join(data_path, data_cfg["train"])
    dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]] if isinstance(data_cfg["dev"], list) else os.path.join(data_path, data_cfg["dev"])
    test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]] if isinstance(data_cfg["test"], list) else os.path.join(data_path, data_cfg["test"])

    pad_feature_size = sum(data_cfg["feature_size"]) if isinstance(data_cfg["feature_size"], list) else data_cfg["feature_size"]
    level = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    def tokenize_text(text):
        return list(text) if level == "char" else text.split()

    def tokenize_features(features):
        return [ft.squeeze() for ft in torch.split(features, 1, dim=0)]

    sequence_field = data.RawField()
    signer_field = data.RawField()
    sgn_field = data.Field(
        use_vocab=False, init_token=None, dtype=torch.float32, 
        preprocessing=tokenize_features, tokenize=lambda features: features, 
        batch_first=True, include_lengths=True,
        pad_token=torch.zeros((pad_feature_size,))
    )
    gls_field = data.Field(
        pad_token=PAD_TOKEN, tokenize=tokenize_text,
        batch_first=True, lower=False, include_lengths=True
    )
    txt_field = data.Field(
        init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
        tokenize=tokenize_text, unk_token=UNK_TOKEN, 
        batch_first=True, lower=txt_lowercase, include_lengths=True
    )

    train_data = SignTranslationDataset(
        path=train_paths, fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
        filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length and len(vars(x)["txt"]) <= max_sent_length,
    )

    # Mostrar muestras del dataset para inspección
    print(f"\nMostrando {num_samples} muestras de los datos de entrenamiento:\n")
    for i, sample in enumerate(train_data[:num_samples]):
        print(f"Muestra {i + 1}:")
        for key, value in vars(sample).items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: Tensor con forma {value.shape}")
            else:
                print(f"{key}: {value}")
        print("\n" + "-" * 40 + "\n")

    # Cargar vocabularios y dataset de desarrollo y prueba
    gls_vocab = build_vocab("gls", min_freq=data_cfg.get("gls_voc_min_freq", 1), 
                            max_size=data_cfg.get("gls_voc_limit", sys.maxsize), 
                            dataset=train_data, vocab_file=data_cfg.get("gls_vocab"))
    txt_vocab = build_vocab("txt", min_freq=data_cfg.get("txt_voc_min_freq", 1), 
                            max_size=data_cfg.get("txt_voc_limit", sys.maxsize), 
                            dataset=train_data, vocab_file=data_cfg.get("txt_vocab"))

    dev_data = SignTranslationDataset(
        path=dev_paths, fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field)
    )
    test_data = SignTranslationDataset(
        path=test_paths, fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field)
    )

    gls_field.vocab = gls_vocab
    txt_field.vocab = txt_vocab
    return train_data, dev_data, test_data, gls_vocab, txt_vocab

if __name__ == '__main__':
    load_data(, 1)