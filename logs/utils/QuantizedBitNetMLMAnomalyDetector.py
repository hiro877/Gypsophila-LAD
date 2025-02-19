# main.py
import argparse
import os
import sys

from .models.anomaly_detection.parameter.model_utils import MaskedTrainDataset, MaskedTextTestDataset, ModelTrainerForBitNet, ModelTester, DataHandler
from torch.utils.data import DataLoader
import torch
import json
torch.cuda.empty_cache()
from .models.QuantizedBitNetForMaskedLM import BitNetForMaskedLM, BitNetConfig
from torch.optim import AdamW
from tokenizers import BertWordPieceTokenizer



def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


import os


def list_file_paths(directory):
    """
    指定されたディレクトリ内のすべての.pthファイルのパスをリストとして返す関数。

    Args:
    directory (str): ファイルパスを取得するディレクトリのパス。

    Returns:
    list: ディレクトリ内のすべての.pthファイルのフルパスのリスト。
    """
    # .pthファイルのパスを格納するためのリスト
    pth_file_paths = []

    # ディレクトリ内のすべてのファイルとフォルダを取得
    for root, dirs, files in os.walk(directory):
        for file in files:
            # .pthファイルのみをリストに追加
            if file.endswith('.pth'):
                pth_file_paths.append(os.path.join(root, file))

    pth_file_paths.sort()
    return pth_file_paths


def extract_sort_keys_precise(file_path):
    parts = file_path.split('/')
    base_name = parts[-1].split('.')[0]  # e.g., saved_model_1000000_10
    model_size = int(base_name.split('_')[2])  # Extract the model size (100000, 1000000)
    version = base_name.split('_')[-1]  # Extract the version part, which might be the model size itself

    # If version is not a number, it's the base file, set a high sort value to place it last in its group
    if version.isdigit():
        version_number = int(version)
    else:
        version_number = float('inf')  # This ensures the base file without suffix goes to the end

    return (model_size, version_number)



def train_quantized_bitnet(train_data_path, tokenizer_file):

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default="config.json", type=str, help="Path to the configuration file")

    config_path = "./logs/utils/configurations/exp3.json"
    params = load_config(config_path)
    print("params: \n", params)

    # 1) Tokenizer のロード
    tokenizer = BertWordPieceTokenizer(tokenizer_file)
    vocab_size_ = tokenizer.get_vocab_size()
    print("Use Vocab Size is ", vocab_size_)

    # 2) コンフィグ
    config = BitNetConfig(
        vocab_size=vocab_size_,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        bit_width=8  # 量子化ビット幅
    )

    # 3) デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 4) モデル初期化
    model = BitNetForMaskedLM(config).to(device)

    # 5) オプティマイザ
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"])

    # 6) Dataset / DataLoader
    train_dataset = MaskedTrainDataset(
        file_path=train_data_path,
        tokenizer=tokenizer,
        use_proposed_method=params["use_proposed_method"],
        learn_positinal_info=params["learn_positinal_info"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=DataHandler.collate_batch
    )

    # 7) Trainer (例: ModelTrainer)
    trainer = ModelTrainerForBitNet(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        device=device,
        model_path=params['saved_model_dir'],
        epochs=params["epochs"]
    )
    if params["load_model_path"] is not None:
        trainer.load_model(params["load_model_path"], config)

    # 8) 学習実行
    trainer.train()

def test_(test_data_path):
    pass


if __name__ == '__main__':
    train_quantized_bitnet()
