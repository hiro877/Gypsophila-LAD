# main.py
import argparse
import os
import sys

from .models.anomaly_detection.parameter.model_utils import MaskedTrainDataset, MaskedTextTestDataset, ModelTrainer, ModelTester, DataHandler
from torch.utils.data import DataLoader
import torch
import json
torch.cuda.empty_cache()

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



def train_(train_data_path, tokenizer_file):

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default="config.json", type=str, help="Path to the configuration file")

    config_path = "./logs/utils/configurations/exp3.json"
    params = load_config(config_path)
    print("params: \n", params)
    # args = parser.parse_args()
    # sys.exit()

    """ Tokenizerの準備 """
    # トークナイザーの初期化
    tokenizer = BertWordPieceTokenizer(tokenizer_file)

    if params["use_proposed_method"]:
        # TODO:とりあえずの値を使用中...
        vocab_size_ = 200000
    else:
        vocab_size_=tokenizer.get_vocab_size()
    print("Use Vocab Size is ", tokenizer.get_vocab_size())
    # TODO: パラメータの設定を考える
    # BitNet用の設定を行う
    config = BitNetConfig(
        vocab_size=vocab_size_,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512
    )

    # デバイスの設定（GPUがあればGPU、なければCPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # BitNetモデルの初期化（MaskedLMタスク用）
    model = BitNetForMaskedLM(config).to(device)

    # オプティマイザーの設定
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"])

    train_dataset = MaskedTrainDataset(train_data_path, tokenizer, params["use_proposed_method"], params["learn_positinal_info"])
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, collate_fn=DataHandler.collate_batch)

    trainer = ModelTrainer(model, train_loader, optimizer, device, params['saved_model_dir'], params["epochs"])
    if params["load_model_path"] is not None:
        trainer.load_model(params["load_model_path"], config)
    trainer.train()

def test_(test_data_path):
    pass


if __name__ == '__main__':
    train_()
