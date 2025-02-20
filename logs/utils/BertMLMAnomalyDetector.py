# main.py
import argparse
import os
import sys

from .models.anomaly_detection.parameter.model_utils import MaskedTrainDataset, MaskedTextTestDataset, ModelTrainer, ModelTester, DataHandler
from transformers import BertForMaskedLM, AdamW, BertConfig
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
import torch
import json
torch.cuda.empty_cache()
# 必要に応じて、他のインポートも追加してください
import logging
logging.basicConfig(level=logging.INFO)
# ロガーの設定
logger = logging.getLogger(__name__)


"""
python main_param_ad.py --epochs 50 --vocab_size 100 --train_data_num 10000 --param_state Info
python main_param_ad.py --epochs 50 --vocab_size 100 --train_data_num 10000 --param_state Info --only_test
python main_param_ad.py --epochs 50 --vocab_size 100 --train_data_num 10000 --param_state Value --only_test

python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 10000 --param_state Info
python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 10000 --param_state Info --saved_model_dir saved_models/20000 --only_test
python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 10000 --param_state Value --saved_model_dir saved_models/20000 --only_test --thre_AD 0.00015

python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 100000 --param_state Info
python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 100000 --param_state Info --saved_model_dir saved_models/20000/data_num_100000 --only_test --thre_AD 0.00015

### Improved Methods
python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 10000 --param_state Info
python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 10000 --param_state Info --saved_model_dir saved_models/improved_method/20000/data_num_100000 --only_test --thre_AD 0.00015
python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 10000 --param_state Value --saved_model_dir saved_models/improved_method/20000/data_num_10000 --only_test --thre_AD 0.00015

python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 100000 --param_state Info
python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 100000 --param_state Value --saved_model_dir saved_models/improved_method/20000/data_num_100000 --only_test --thre_AD 0.00015

- Load Model
python main_param_ad.py --epochs 50 --vocab_size 20000 --train_data_num 10000 --param_state Info --load_model_path saved_models/improved_method/20000/data_num_10000/50.pth

"""

"""
python main_param_ad.py --config configurations/config.json
"""

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



def test(params):
    tokenizer_file = f'models/anomaly_detection/parameter/trained_tokenizer/{params["tokenizer_dir"]}/vocab.txt'
    tokenizer = BertWordPieceTokenizer(tokenizer_file)
    if params["use_proposed_method"]:
        vocab_size_ = 200000
    else:
        vocab_size_=tokenizer.get_vocab_size()
    print("Use Vocab Size is ", tokenizer.get_vocab_size())
    config = BertConfig(vocab_size=vocab_size_, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM(config).to(device)

    test_data_path = f"datasets_for_models/sample_param/test/{params['test_data_dir']}dataset_test_{params['param_state'].lower()}.txt"
    test_dataset = MaskedTextTestDataset(test_data_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    # directory_path = "saved_models/{}".format(params["tokenizer_dir"])
    directory_path = params['saved_model_dir']
    saved_model_files = list_file_paths(directory_path)
    # saved_model_files = sorted(saved_model_files, key=extract_sort_keys_precise)
    print("saved_model_files: ", saved_model_files)

    tester = ModelTester(model, tokenizer, test_loader, params['use_proposed_method'], device, params['is_analyzing'])
    # eval_results = tester.test(params["param_state"], params["thre_AD"], "results/miss_result.txt")
    # print("Evaluation Results:", eval_results)

    # filename = "RESULTS_TEST_Value_CognitiveRobotics.txt"
    filename = "RESULTS_TEST_ICEE2025.txt"
    results_file_path = "results/" + filename
    command_line_string = " ".join(sys.argv)
    with open(results_file_path, "a+") as fw:
        fw.write("\n" + "==========\n" + command_line_string + "\n\n")
        file_num=1
        for model_path in saved_model_files:
            if params["is_analyzing"]:
                if model_path != "saved_models/exp23w/0050.pth":
                    continue
            # if model_path != "saved_models/exp23w/0050.pth":
            #     continue

            print("Evaluste file={}".format(model_path))
            tester.load_model(model_path, config)

            # print(model)
            # sys.exit()

            # eval_results = test(model, data_loader, device, params["param_state"], params["thre_AD"])
            eval_results = tester.test(params["param_state"], params["thre_AD"], "results/miss_result.txt")

            result = f"{eval_results['f1']} {eval_results['false_alarm_rate']} {eval_results['underreport_rate']} {eval_results['rc']} {eval_results['pc']} {eval_results['acc']} " \
                   f"{eval_results['tn']} {eval_results['fp']} {eval_results['fn']} {eval_results['tp']}"


            fw.write(result+"\n")
            if file_num % 8 == 0:
                fw.write("\n")
            file_num+=1

        for model_path in saved_model_files:
            fw.write(model_path + "\n")


def only_analyze_predictions(params):
    tokenizer_file = f'models/anomaly_detection/parameter/trained_tokenizer/{params["tokenizer_dir"]}/vocab.txt'
    print(tokenizer_file)
    tokenizer = BertWordPieceTokenizer(tokenizer_file)
    if params["use_proposed_method"]:
        vocab_size_ = 200000
    else:
        vocab_size_=tokenizer.get_vocab_size()
    print("Use Vocab Size is ", tokenizer.get_vocab_size())
    config = BertConfig(vocab_size=vocab_size_, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM(config).to(device)

    test_data_path = f"datasets_for_models/sample_param/test/dataset_test_{params['param_state'].lower()}.txt"
    test_dataset = MaskedTextTestDataset(test_data_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)


    tester = ModelTester(model, tokenizer, test_loader, params['use_proposed_method'], device)

    texts=["state=B, value=[MASK], InfoD", "state=C, value=[MASK], InfoN", "state=C, value=[MASK], InfoD", "state=A, value=[MASK], InfoA"]
    masked_params=["1743", "1000", "46", "/", "2327", "1813", "1667", "1232", "74"]
    labels=["-", "-", "Anomaly", "/", "-", "-", "-", "-", "-"]

    """
    Settomg by Hand
    """
    texts_ind = 0
    targed_ind = 0
    target_exp = "exp3_0125x1"
    target_model_num = "0020"

    model_path = "saved_models/{}/{}.pth".format(target_exp, target_model_num)
    text, masked_param, label = texts[texts_ind], masked_params[targed_ind], labels[targed_ind]
    print("Evaluste file={}".format(model_path))
    tester.load_model(model_path, config)
    tester.only_analyze_ditections(params["param_state"], params["thre_AD"], text, masked_param, label)

"""
Settomg Description
0: Exp7 Miss
1: Exp7 Correct (value=80~2501)
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    params = load_config(args.config)
    print("params: \n", params)
    # sys.exit()
    if(params["only_test"]):
        if (params["analyze_predictions"]):
            only_analyze_predictions(params)
            return
        test(params)
        return


    tokenizer_file = f'models/anomaly_detection/parameter/trained_tokenizer/{params["tokenizer_dir"]}/vocab.txt'
    tokenizer = BertWordPieceTokenizer(tokenizer_file)

    if params["use_proposed_method"]:
        vocab_size_ = 200000
    else:
        vocab_size_=tokenizer.get_vocab_size()
    print("Use Vocab Size is ", tokenizer.get_vocab_size())
    config = BertConfig(vocab_size=vocab_size_, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # sys.exit()
    model = BertForMaskedLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"])

    train_data_path = f"datasets_for_models/sample_param/train/{params['train_data_path']}.txt"
    train_dataset = MaskedTextDataset(train_data_path, tokenizer, params["use_proposed_method"], params["learn_positinal_info"])
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, collate_fn=DataHandler.collate_batch)

    test_data_path = f"datasets_for_models/sample_param/test/{params['test_data_dir']}dataset_test_{params['param_state'].lower()}.txt"
    test_dataset = MaskedTextTestDataset(test_data_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    trainer = ModelTrainer(model, train_loader, optimizer, device, params['saved_model_dir'], params["epochs"])
    if params["load_model_path"] is not None:
        trainer.load_model(params["load_model_path"], config)
    trainer.train()

    tester = ModelTester(model, tokenizer, test_loader, params['use_proposed_method'], device)
    eval_results = tester.test(params["param_state"], params["thre_AD"], "results/miss_result.txt")
    print("Evaluation Results:", eval_results)


# ※ MaskedTrainDataset と DataHandler は適宜インポートしてください
# from your_module import MaskedTrainDataset, DataHandler
# また、load_config() も適切にインポートしてください
import time
import torch
from transformers import BertForMaskedLM, BertConfig, AdamW
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

# ※ MaskedTrainDataset、DataHandler、load_config などは適宜インポートしてください
# from your_module import MaskedTrainDataset, DataHandler, load_config

def train_(train_data_path, tokenizer_file):
    overall_start = time.time()  # 全体の開始時間

    # 設定ファイルのロード
    config_path = "./logs/utils/configurations/exp3.json"
    params = load_config(config_path)
    print("params: \n", params)

    # Tokenizerの準備
    t0 = time.time()
    tokenizer = BertWordPieceTokenizer(tokenizer_file)
    t_tokenizer = time.time() - t0
    print(f"Tokenizer loaded in {t_tokenizer:.2f} seconds.")

    # 語彙サイズの決定と設定の作成
    t1 = time.time()
    if params["use_proposed_method"]:
        vocab_size_ = 200000
    else:
        vocab_size_ = tokenizer.get_vocab_size()
    print("Use Vocab Size is ", tokenizer.get_vocab_size())
    config = BertConfig(
        vocab_size=vocab_size_,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512
    )
    t_config = time.time() - t1
    print(f"Configuration set in {t_config:.2f} seconds.")

    # デバイス、モデル、オプティマイザのセットアップ
    t2 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = BertForMaskedLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"])
    t_model = time.time() - t2
    print(f"Model and optimizer created in {t_model:.2f} seconds.")

    # データセットとデータローダの準備
    t3 = time.time()
    train_dataset = MaskedTrainDataset(
        train_data_path,
        tokenizer,
        params["use_proposed_method"],
        params["learn_positinal_info"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=DataHandler.collate_batch
    )
    t_data = time.time() - t3
    print(f"Dataset and DataLoader created in {t_data:.2f} seconds.")

    # Trainerの作成と、必要ならモデルのロード
    t4 = time.time()
    trainer = ModelTrainer(
        model, train_loader, optimizer, device,
        params['saved_model_dir'], params["epochs"]
    )
    if params["load_model_path"] is not None:
        trainer.load_model(params["load_model_path"], config)
    t_trainer = time.time() - t4
    print(f"Trainer set up in {t_trainer:.2f} seconds.")

    # 全体の前処理時間を計測
    pre_train_total = time.time() - overall_start
    print(f"Total time before calling trainer.train(): {pre_train_total:.2f} seconds.")

    # Channelsのグループに計測結果を送信（Webブラウザに表示される）
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "training_progress",
        {
            "type": "training_progress",
            "message": (
                f"Pre-training completed in {pre_train_total:.2f} seconds:\n"
                f" - Tokenizer: {t_tokenizer:.2f} sec\n"
                f" - Config: {t_config:.2f} sec\n"
                f" - Model & Optimizer: {t_model:.2f} sec\n"
                f" - DataLoader: {t_data:.2f} sec\n"
                f" - Trainer setup: {t_trainer:.2f} sec"
            )
        }
    )

    # ここから学習開始
    trainer.train()



def test_(test_data_path, tokenizer_file):
    tokenizer = BertWordPieceTokenizer(tokenizer_file)

    # 設定ファイルのロード
    config_path = "./logs/utils/configurations/exp3.json"
    params = load_config(config_path)
    logger.info(f"load_config: {params}")


    if params["use_proposed_method"]:
        vocab_size_ = 200000
    else:
        vocab_size_ = tokenizer.get_vocab_size()
    logger.info(f"Use Vocab Size is  {tokenizer.get_vocab_size()}")
    config = BertConfig(vocab_size=vocab_size_, hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
                        intermediate_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM(config).to(device)

    test_dataset = MaskedTextTestDataset(test_data_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    directory_path = params['saved_model_dir']
    # saved_model_files = list_file_paths(directory_path)
    saved_model_file = directory_path + "/0001.pth"
    logger.info(f"saved_model_file: {saved_model_file}")


    tester = ModelTester(model, tokenizer, test_loader, params['use_proposed_method'], device,
                         params['is_analyzing'])

    filename = "RESULTS_TEST_ICEE2025.txt"
    results_file_path = "results/" + filename
    os.makedirs("results", exist_ok=True)
    command_line_string = " ".join(sys.argv)
    with open(results_file_path, "a+") as fw:
        fw.write("\n" + "==========\n" + command_line_string + "\n\n")

        logger.info(f"Evaluste file=: {saved_model_file}")
        tester.load_model(saved_model_file, config)

        eval_results = tester.test(params["param_state"], params["thre_AD"], "results/miss_result.txt")
        logger.info(f"eval_results: {eval_results}")

        # result = f"{eval_results['f1']} {eval_results['false_alarm_rate']} {eval_results['underreport_rate']} {eval_results['rc']} {eval_results['pc']} {eval_results['acc']} " \
        #          f"{eval_results['tn']} {eval_results['fp']} {eval_results['fn']} {eval_results['tp']}"
        # fw.write(result + "\n")
    return eval_results


if __name__ == '__main__':
    main()
