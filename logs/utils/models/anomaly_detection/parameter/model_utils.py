import sys
import os
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForMaskedLM, AdamW, BertConfig
from tokenizers import BertWordPieceTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import argparse
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import logging
logging.basicConfig(level=logging.INFO)
# ロガーの設定
logger = logging.getLogger(__name__)

color_dic = {"black":"\033[30m", "red":"\033[31m", "green":"\033[32m", "yellow":"\033[33m", "blue":"\033[34m", "end":"\033[0m"}
def print_color(text, color="red"):
    print(color_dic[color] + text + color_dic["end"])

class AnalysisTools:
    """
    A class for various analysis tools and utilities.
    """
    def __init__(self):
        self.color_dic = {"black": "\033[30m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "blue": "\033[34m", "end": "\033[0m"}

    def print_color(self, text, color="red"):
        """
        Print text in the specified color.
        """
        print(self.color_dic[color] + text + self.color_dic["end"])

    def calc_mad(self, data):
        """
        Calculate the Median Absolute Deviation of data.
        """
        mad = np.median(np.abs(data - np.median(data)))
        return mad

class Metrics:
    """
    A class to calculate evaluation metrics.
    """
    @staticmethod
    def calculate_metrics(pred_list, label_list, zero_div_option=0):
        """
        Calculate precision, recall, F1 score, and accuracy.
        """
        precision = precision_score(label_list, pred_list, average='binary', zero_division=zero_div_option)
        recall = recall_score(label_list, pred_list, average='binary', zero_division=zero_div_option)
        f1 = f1_score(label_list, pred_list, average='binary', zero_division=zero_div_option)
        accuracy = accuracy_score(label_list, pred_list)
        return precision, recall, f1, accuracy

class DataHandler:
    """
    A class to handle data loading and batching.
    """
    @staticmethod
    def collate_batch(batch):
        """
        Collate batch of data for DataLoader.
        """
        input_ids, labels = zip(*batch)
        input_ids_padded = pad_sequence([torch.tensor(seq) for seq in input_ids], batch_first=True, padding_value=0)
        labels_padded = pad_sequence([torch.tensor(lab) for lab in labels], batch_first=True, padding_value=-100)
        return input_ids_padded, labels_padded

class MaskedTrainDataset(Dataset):
    """
    Dataset class for masked text.
    """
    def __init__(self, file_path, tokenizer, use_proposed_method, learn_positinal_info=True, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_proposed_method = use_proposed_method
        self.learn_positinal_info = learn_positinal_info
        with open(file_path, 'r', encoding='utf-8') as file:
            self.texts = [line.strip() for line in file.readlines()]

    def __len__(self):
        return len(self.texts)

    def random_word(self, tokens):
        output_labels = []
        masked_token_indexes = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    tokens[i] = self.tokenizer.token_to_id("[MASK]")
                    masked_token_indexes.append(i)
                elif prob < 0.9:
                    tokens[i] = np.random.randint(0, self.tokenizer.get_vocab_size())
                output_labels.append(token)
            else:
                output_labels.append(-100)
        return tokens, output_labels, masked_token_indexes

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens_  = self.tokenizer.encode(text).ids
        # print(tokens_)
        # # エンコード結果のトークンID列をそれぞれのトークンに変換
        # tokens = self.tokenizer.convert_ids_to_tokens(tokens_)
        # print("各トークンIDと対応するトークン:", tokens)
        # sys.exit()
        # tokens_ = encoding.ids
        tokens, labels, masked_token_indexes = self.random_word(tokens_)
        if self.use_proposed_method:
            tokens, labels = self.add_positional_info(tokens_, labels, masked_token_indexes)

        return torch.tensor(tokens), torch.tensor(labels)

    def add_positional_info(self, tokens, labels, masked_token_indexes):
        # Exclude the first and last elements and the masked token indexes
        filtered_tokens = [tokens[i] for i in range(1, len(tokens) - 1) if i not in masked_token_indexes]
        # Calculate the sum of the filtered tokens
        total_sum = sum(filtered_tokens)
        if self.learn_positinal_info:
            tokens.insert(-1, self.tokenizer.token_to_id("[MASK]"))
            labels.insert(-1, total_sum)
        else:
            tokens.insert(-1, total_sum)
            labels.insert(-1, -100)
        return tokens, labels

class MaskedTextTestDataset(Dataset):
    """
    Dataset class for testing masked text.
    """
    def __init__(self, file_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = []
        self.masked_params = []
        self.labels = []
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines()]
        for line in lines:
            parts = line.split()
            self.masked_params.append(parts[-2])
            self.labels.append(parts[-1])
            self.texts.append(" ".join(parts[:-2]))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        masked_param = self.masked_params[idx]
        label = self.labels[idx]
        return text, masked_param, label


class ModelTrainer:
    """
    A class to handle model training.
    """

    def __init__(self, model, data_loader, optimizer, device, model_path, epochs):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device
        self.model_path = model_path
        self.epochs = epochs
        self.save_epochs = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        os.makedirs(self.model_path, exist_ok=True)
        self.epoch_offset = 1

    def train(self):
        """
        Train the model.
        """
        print("Start Training...")
        self.model.train()
        filename = "loss.log"
        loss_file_path = "{}/{}".format(self.model_path, filename)
        command_line_string = " ".join(sys.argv)

        # チャンネルレイヤの取得
        channel_layer = get_channel_layer()

        with open(loss_file_path, "a+") as fw:
            fw.write(command_line_string + "\n======================\n")
            for epoch in range(self.epochs):
                total_loss = 0
                progress_bar = tqdm(self.data_loader, desc=f"Epoch {epoch + self.epoch_offset}", unit="batch")
                for batch_idx, (tokens, labels) in enumerate(progress_bar):
                    tokens, labels = tokens.to(self.device), labels.to(self.device)
                    self.model.zero_grad()
                    outputs = self.model(tokens, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                    # バッチごとに進捗情報を送信
                    async_to_sync(channel_layer.group_send)(
                        "training_progress",
                        {
                            "type": "training_progress",
                            "message": f"Epoch {epoch + self.epoch_offset}, Batch {batch_idx + 1}/{len(self.data_loader)}: Loss = {loss.item():.4f}"
                        }
                    )
                    progress_bar.set_postfix(loss=loss.item())

                avg_loss = total_loss / len(self.data_loader)
                # エポック終了時に進捗情報を送信
                async_to_sync(channel_layer.group_send)(
                    "training_progress",
                    {
                        "type": "training_progress",
                        "message": f"Epoch {epoch + self.epoch_offset} finished. Average Loss = {avg_loss:.4f}"
                    }
                )

                if (epoch + self.epoch_offset) in self.save_epochs:
                    save_path = "{}/{:04d}.pth".format(self.model_path, epoch + self.epoch_offset)
                    torch.save(self.model.state_dict(), save_path)
                print(f"Epoch {epoch + self.epoch_offset}: Average Loss: {avg_loss}")
                fw.write(f"Epoch {epoch + self.epoch_offset}: Average Loss: {avg_loss}\n")

    def load_model(self, path, config):
        print("Loading model...")
        print("path = {}".format(path))
        self.model = BertForMaskedLM(config)
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

        epoch_offset_ = self.epoch_offset
        self.epoch_offset = int(path.split("/")[-1].split(".")[0]) + self.epoch_offset
        self.save_epochs = [x + (self.epoch_offset - epoch_offset_) for x in self.save_epochs]
        print(self.save_epochs)
        print(self.epoch_offset)
        print("Re Training From Epoch = {}".format(self.epoch_offset))

class ModelTester:
    """
    A class to handle model testing and evaluation.
    """
    def __init__(self, model, tokenizer, data_loader, use_proposed_method, device, is_analyzing=False):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.device = device
        self.is_analyzing = is_analyzing
        self.anomaly_calculator = AnomalyCalculator(tokenizer)
        self.use_proposed_method = use_proposed_method

    def load_model(self, path, config):
        self.model = BertForMaskedLM(config)
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def test(self, param_state, thre_AD, miss_results_file_path):
        """
        Test the model and calculate evaluation metrics.
        """
        self.model.eval()
        text_list, pred_list = [], []
        os.makedirs(os.path.dirname(miss_results_file_path), exist_ok=True)
        line_num = 1
        with open(miss_results_file_path, 'a+') as file:
            for text_batch, masked_param_batch, label_batch in self.data_loader:
                for i in range(len(text_batch)):
                    logger.info(f"line_num: {line_num}")
                    text, masked_param, label = text_batch[i], masked_param_batch[i], label_batch[i]
                    encoded_input = self.tokenizer.encode(text).ids
                    mask_index = encoded_input.index(self.tokenizer.token_to_id("[MASK]"))

                    input_ids = torch.tensor([encoded_input]).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        predictions = outputs.logits

                    score, _score = self.anomaly_calculator.calc_anomaly(param_state, masked_param, predictions, mask_index)

                    pred = 1 if score < thre_AD else 0
                    pred_list.append(pred)
                    text_list.append(text)

                    if self.is_analyzing:
                        self.analyze_predictions(pred, label, score, text, predictions, mask_index, param_state, masked_param, thre_AD)

                    line_num+=1


        eval_results = {
            "text_list": text_list,
            "pred_list": pred_list,
        }
        return eval_results

    def test_exp(self, param_state, thre_AD, miss_results_file_path):
        """
        Test the model and calculate evaluation metrics.
        """
        self.model.eval()
        label_list, pred_list = [], []
        os.makedirs(os.path.dirname(miss_results_file_path), exist_ok=True)

        with open(miss_results_file_path, 'a+') as file:
            for text_batch, masked_param_batch, label_batch in self.data_loader:
                for i in range(len(text_batch)):
                    text, masked_param, label = text_batch[i], masked_param_batch[i], label_batch[i]
                    encoded_input = self.tokenizer.encode(text).ids
                    mask_index = encoded_input.index(self.tokenizer.token_to_id("[MASK]"))

                    input_ids = torch.tensor([encoded_input]).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        predictions = outputs.logits

                    score, _score = self.anomaly_calculator.calc_anomaly(param_state, masked_param, predictions, mask_index)

                    pred = 1 if score < thre_AD else 0
                    pred_list.append(pred)
                    label = 0 if label == "-" else 1
                    label_list.append(label)

                    if self.is_analyzing:
                        self.analyze_predictions(pred, label, score, text, predictions, mask_index, param_state, masked_param, thre_AD)

        # print("1: precision={}, recall={}, f1={}, accuracy={}".format(precision, recall, f1, accuracy))
        precision, recall, f1, accuracy = calculate_metrics(pred_list, label_list)
        #print("2: precision={}, recall={}, f1={}, accuracy={}".format(precision, recall, f1, accuracy))
        cm = confusion_matrix(label_list, pred_list)
        tn, fp, fn, tp = cm.flatten()

        false_alarm_rate = fp / (fp + tn) if (fp + tn) != 0 else 0
        underreport_rate = fn / (fn + tp) if (fn + tp) != 0 else 0
        eval_results = {
            "f1": f1,
            "false_alarm_rate": false_alarm_rate,
            "underreport_rate": underreport_rate,
            "rc": recall,
            "pc": precision,
            "acc": accuracy,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        return eval_results

    # def add_positional_info(self, tokens, mask_index):
    #     # Exclude the first and last elements and the masked token indexes
    #     filtered_tokens = [tokens[i] for i in range(1, len(tokens) - 1) if i not in [mask_index]]
    #     # Calculate the sum of the filtered tokens
    #     total_sum = sum(filtered_tokens)
    #
    #     tokens.insert(-1, total_sum)
    #     return tokens

    def analyze_predictions(self, pred, label, score, text, predictions, mask_position, param_state, masked_param, thre_AD):
        print("analyze_predictions...", param_state)
        if param_state == "State":
            self.analyze_state(pred, label, score, text, predictions, mask_position, param_state, masked_param, thre_AD)
        elif param_state == "Value":
            self.analyze_value(pred, label, score, text, predictions, mask_position, param_state, masked_param, thre_AD)
        # elif param_state == "Info":
        #     break
        #
        # param_type = ["A", "B", "C"]
        # pt_i = 0
        # if pred == label and text[6] == param_type[pt_i] and label==0:
        #     print("text", text, text[6])
        #     # print("encoded_input", encoded_input)
        #     print("predictions", predictions.shape)
        #
        #     id_best = predictions[0, mask_position].argmax(-1).item()
        #     token_best = self.tokenizer.id_to_token(id_best)
        #     # if token_best is None: return
        #     token_best = token_best.replace("##", "")
        #     text = text.replace("[MASK]", token_best)
        #
        #     print_color("[Miss]: pred={}, label={}, score={}".format(pred, label, score))
        #     print("text={}, token_best={}, masked_param={}".format(text, token_best, masked_param))
        #     self.anomaly_calculator.calc_anomaly(param_state, masked_param.lower(), predictions, mask_position, True)
        #     print("=" * 20)
        #     """ ============================================== """
        #     all_probability = 0
        #     prob_list = []
        #     for ind in range(10001):
        #         encoding = self.tokenizer.encode(str(ind), add_special_tokens=False)
        #         info_b_token_id = encoding.ids[0]  # .ids でIDのリストを取得し、その最初の要素を選択
        #         probability = torch.nn.functional.softmax(predictions[0, mask_position], dim=-1)[info_b_token_id].item()
        #         print("{} {}".format(ind, format(probability, ".20f")))
        #         all_probability += probability
        #         prob_list.append(probability)
        #
        #     self.save_prob(prob_list, [80, 2500], thre_AD, "result{}.png".format(param_type[pt_i]))
        #     print(1 / 12380)
        #     per1 = 1 / 12380
        #     print(format(per1, ".20f"))
        #     print(format(30 * per1, ".20f"))
        #     print("all_probability {}".format(all_probability))
        #     sys.exit()
    def analyze_state(self, pred, label, score, text, predictions, mask_position, param_state, masked_param, thre_AD):
        param_type = ["A", "B", "C"]
        pt_i = 0
        if pred != label:
            print("text", text, masked_param)
            # print("predictions", predictions.shape)

            id_best = predictions[0, mask_position].argmax(-1).item()
            token_best = self.tokenizer.id_to_token(id_best)
            token_best = token_best.replace("##", "")
            text = text.replace("[MASK]", token_best)

            # print_color("[Miss]: pred={}, label={}, score={}".format(pred, label, score))
            # print("text={}, token_best={}, masked_param={}".format(text, token_best, masked_param))
            # print("=" * 20)
            """ ============================================== """
            all_probability = 0
            prob_list = []
            vocab_size_ = self.tokenizer.get_vocab_size()
            for token_id in range(vocab_size_):
                probability = torch.nn.functional.softmax(predictions[0, mask_position], dim=-1)[token_id].item()
                # print("token_id={} probability={}".format(token_id, format(probability, ".20f")))
                all_probability += probability
                prob_list.append(probability)

            # self.save_prob_state(prob_list, [0, vocab_size_], thre_AD, "result{}.png".format(param_type[pt_i]))
            # print("vocab_size_: ", vocab_size_, "1/vocab_size_ = ", 1 / vocab_size_)
            # per1 = 1 / vocab_size_
            # print(format(per1, ".20f"))
            # print("all_probability {}".format(all_probability))
            # sys.exit()


    def analyze_value(self, pred, label, score, text, predictions, mask_position, param_state, masked_param, thre_AD):
        param_type = ["A", "B", "C"]
        # param_type = ["U", "C", "I", "D", "A"]
        pt_i = 1
        print("analyze_value ...")
        # if pred == label and text[6] == param_type[pt_i] and label == 0:
        if True:
            print("text", text, text[6])
            # print("encoded_input", encoded_input)
            print("predictions", predictions.shape)

            id_best = predictions[0, mask_position].argmax(-1).item()
            token_best = self.tokenizer.id_to_token(id_best)
            # if token_best is None: return
            token_best = token_best.replace("##", "")
            text = text.replace("[MASK]", token_best)

            print_color("[Miss]: pred={}, label={}, score={}".format(pred, label, score))
            print("text={}, token_best={}, masked_param={}".format(text, token_best, masked_param))
            # self.anomaly_calculator.calc_anomaly(param_state, masked_param.lower(), predictions, mask_position, True)
            self.anomaly_calculator.calc_anomaly2(param_state, masked_param.lower(), predictions, mask_position, True)
            print("=" * 20)
            """ ============================================== """
            all_probability = 0
            prob_list = []
            for ind in range(10001):
                encoding = self.tokenizer.encode(str(ind), add_special_tokens=False)
                info_b_token_id = encoding.ids[0]  # .ids でIDのリストを取得し、その最初の要素を選択
                probability = torch.nn.functional.softmax(predictions[0, mask_position], dim=-1)[info_b_token_id].item()
                # print("{} {}".format(ind, format(probability, ".20f")))
                all_probability += probability
                prob_list.append(probability)
                # sys.exit()


            self.save_prob(prob_list, [80, 2501], thre_AD, "result{}.png".format(param_type[pt_i]), y_lim=0.002)
            print(1 / 12380)
            per1 = 1 / 12380
            print(format(per1, ".20f"))
            print(format(30 * per1, ".20f"))
            print("all_probability {}".format(all_probability))
            sys.exit()

    def save_prob(self, probabilities, sub_x_datas, sub_y, save_path, y_lim=1):
        print("save_prob()")
        probabilities=probabilities[:5600]
        # X軸のデータ（インデックス）
        x_data = list(range(1, len(probabilities) + 1))

        # 散布図をプロット
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, probabilities, alpha=0.5)
        # for x_value in sub_x_datas:
        plt.axvline(x=sub_x_datas[0], color='r', linestyle='-', linewidth=2, label="x={}".format(sub_x_datas[0]))
        plt.axvline(x=sub_x_datas[1], color='r', linestyle='--', linewidth=2, label="x={}".format(sub_x_datas[1]))

        # plt.axvline(x=500, color='g', linestyle='-', linewidth=2, label="x={}".format(500))
        # plt.axvline(x=1500, color='g', linestyle='--', linewidth=2, label="x={}".format(1500))
        plt.axhline(y=sub_y, color='blue', linestyle='--', label="y={}".format(sub_y))

        # Y軸の範囲を設定
        plt.ylim(0, y_lim)  # 最小値を0、最大値を1.0に設定
        plt.legend(fontsize=14)

        plt.title('Prediction Scores', fontsize=20)
        plt.xlabel('Value', fontsize=16)
        plt.ylabel('Prediction Score', fontsize=16)
        plt.tick_params(axis='both', labelsize=14)  # X軸とY軸両方の目盛りラベルのサイズを12に設定
        print(save_path)
        plt.savefig(save_path)
        print("save prob: {}".format(save_path))

    def save_prob_state(self, probabilities, sub_x_datas, sub_y, save_path):
        # X軸のデータ（インデックス）
        x_data = list(range(1, len(probabilities) + 1))

        # 散布図をプロット
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, probabilities, alpha=0.5)
        # for x_value in sub_x_datas:
        plt.axvline(x=sub_x_datas[0], color='r', linestyle='-', linewidth=2, label="x={}".format(sub_x_datas[0]))
        plt.axvline(x=sub_x_datas[1], color='r', linestyle='--', linewidth=2, label="x={}".format(sub_x_datas[1]))

        # plt.axvline(x=500, color='g', linestyle='-', linewidth=2, label="x={}".format(500))
        # plt.axvline(x=1500, color='g', linestyle='--', linewidth=2, label="x={}".format(1500))
        # plt.axhline(y=sub_y, color='blue', linestyle='--', label="y={}".format(sub_y))

        plt.legend(fontsize=14)

        plt.title('Prediction Scores', fontsize=20)
        plt.xlabel('Value', fontsize=16)
        plt.ylabel('Prediction Score', fontsize=16)
        plt.tick_params(axis='both', labelsize=14)  # X軸とY軸両方の目盛りラベルのサイズを12に設定

        plt.savefig(save_path)


    def only_analyze_ditections(self, param_state, thre_AD, text, masked_param, label):
        """
        Test the model and calculate evaluation metrics.
        """
        self.model.eval()

        text, masked_param, label
        encoded_input = self.tokenizer.encode(text).ids
        mask_index = encoded_input.index(self.tokenizer.token_to_id("[MASK]"))

        input_ids = torch.tensor([encoded_input]).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = outputs.logits

        score, _score = self.anomaly_calculator.calc_anomaly(param_state, masked_param, predictions, mask_index)

        pred = 1 if score < thre_AD else 0
        label = 0 if label == "-" else 1

        self.analyze_predictions(pred, label, score, text, predictions, mask_index, param_state, masked_param, thre_AD)
        if pred != label:
            print("pred, label: ", pred, label)
            print("text: ", text)
            print("mask_index, masked_param: ", mask_index, masked_param)
            sys.exit()
        return


import torch
import numpy as np

class AnomalyCalculator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def calc_topk(self, predictions, mask_index):
        # MASK位置の確率をsoftmaxを使って計算
        softmax = torch.nn.Softmax(dim=-1)
        mask_token_logits = predictions[0, mask_index]
        mask_token_probs = softmax(mask_token_logits)

        # 確率が高い上位5トークンを表示
        top_k = torch.topk(mask_token_probs, 5, dim=-1)
        top_k_probs, top_k_indices = top_k.values, top_k.indices

        # トークンと確率の表示
        for prob, idx in zip(top_k_probs, top_k_indices):
            token = self.tokenizer.id_to_token(idx.item())
            print(f"Token: {token}, Probability: {prob.item()}")

    def calc_value(self, predictions, masked_param, mask_index, is_print=False):
        encoding = self.tokenizer.encode(masked_param, add_special_tokens=False)
        info_b_token_id = encoding.ids[0]
        probability = torch.nn.functional.softmax(predictions[0, mask_index], dim=-1)[info_b_token_id].item()
        if is_print: print(probability)
        return probability

    def calc_all_info(self, predictions, mask_index, param_info, is_print=False):
        probabilities = []
        for param in param_info:
            encoding = self.tokenizer.encode(param, add_special_tokens=False)
            info_b_token_id = encoding.ids[0]
            info_b_probability = torch.nn.functional.softmax(predictions[0, mask_index], dim=-1)[info_b_token_id].item()
            probabilities.append(info_b_probability)

        probabilities = np.array(probabilities)
        return self.normalize_probabilities(probabilities, is_print)

    def calc_anomaly(self, param_state, masked_param, predictions, mask_index, is_print=False):
        if param_state == "State":
            param_info_list = ['A', 'B', 'C']
            pred2index = {"A": 0, "B": 1, "C": 2}
            probabilities = self.calc_all_info(predictions, mask_index, param_info_list, is_print)
            return probabilities[pred2index[masked_param]], probabilities

        elif param_state == "Value":
            probability = self.calc_value(predictions, masked_param, mask_index, is_print)
            return probability, None

        elif param_state == "Info":
            param_info_list = ['InfoA', 'InfoB', 'InfoC', 'InfoD']
            pred2index = {"InfoA": 0, "InfoB": 1, "InfoC": 2, "InfoD": 3}
            probabilities = self.calc_all_info(predictions, mask_index, param_info_list, is_print)
            return probabilities[pred2index[masked_param]], probabilities

        else:
            return None, None

    def calc_anomaly2(self, param_state, masked_param, predictions, mask_index, is_print=False):
        if param_state == "State":
            param_info_list = ['U', 'C', 'I', 'D', 'A']
            pred2index = {"U": 0, "C": 1, "I": 2, "D": 3, "A": 4}
            probabilities = self.calc_all_info(predictions, mask_index, param_info_list, is_print)
            return probabilities[pred2index[masked_param]], probabilities

        elif param_state == "Value":
            probability = self.calc_value(predictions, masked_param, mask_index, is_print)
            return probability, None

        elif param_state == "Info":
            param_info_list = ['InfoH', 'InfoI', 'InfoR', 'InfoN', 'InfoO']
            pred2index = {"InfoH": 0, "InfoI": 1, "InfoR": 2, "InfoN": 3, "InfoO": 4}
            probabilities = self.calc_all_info(predictions, mask_index, param_info_list, is_print)
            return probabilities[pred2index[masked_param]], probabilities

        else:
            return None, None

    def normalize_probabilities(self, probabilities, is_print=False):
        total = sum(probabilities)
        normalized_probabilities = [p / total for p in probabilities]
        if is_print: print("normalized_probabilities: ", normalized_probabilities)
        return normalized_probabilities

def calculate_metrics(pred_list, label_list, zero_div_option=0):
    # 精度（Precision）
    precision = precision_score(label_list, pred_list, average='binary', zero_division=zero_div_option)

    # 再現率（Recall）
    recall = recall_score(label_list, pred_list, average='binary', zero_division=zero_div_option)

    # F1スコア
    f1 = f1_score(label_list, pred_list, average='binary', zero_division=zero_div_option)

    # 正解率（Accuracy）
    accuracy = accuracy_score(label_list, pred_list)

    return precision, recall, f1, accuracy



"""
BitNet
"""

# 量子化版モデルをインポート (例: from quantized_bert import BitNetForMaskedLM)
# from quantized_bert import BitNetForMaskedLM
from ....models.QuantizedBitNetForMaskedLM import BitNetForMaskedLM

class ModelTrainerForBitNet:
    """
    A class to handle model training for the quantized BitNetForMaskedLM model.
    """
    def __init__(self, model, data_loader, optimizer, device, model_path, epochs):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device
        self.model_path = model_path
        self.epochs = epochs

        self.save_epochs = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        os.makedirs(self.model_path, exist_ok=True)
        self.epoch_offset = 1  # 途中再開用のオフセット

    def train(self):
        """
        Train the quantized BitNetForMaskedLM model.
        """
        print("Start Training (BitNetForMaskedLM)...")
        self.model.train()

        filename = "loss.log"
        loss_file_path = os.path.join(self.model_path, filename)

        command_line_string = " ".join(sys.argv)
        with open(loss_file_path, "a+", encoding="utf-8") as fw:
            fw.write(command_line_string + "\n======================\n")

            for epoch in range(self.epochs):
                total_loss = 0.0

                for tokens, labels in self.data_loader:
                    tokens, labels = tokens.to(self.device), labels.to(self.device)
                    self.model.zero_grad()

                    # 量子化モデルの forward
                    outputs = self.model(input_ids=tokens, labels=labels)
                    loss = outputs["loss"]

                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(self.data_loader)

                # 必要なエポックでモデル保存
                current_epoch = epoch + self.epoch_offset
                if current_epoch in self.save_epochs:
                    save_path = "{}/{:04d}.pth".format(self.model_path, current_epoch)
                    torch.save(self.model.state_dict(), save_path)

                print(f"Epoch {current_epoch}: Average Loss: {avg_loss}")
                fw.write(f"Epoch {current_epoch}: Average Loss: {avg_loss}\n")

    def load_model(self, path, config):
        """
        Load the BitNetForMaskedLM model state.
        """
        print("Loading BitNetForMaskedLM model...")
        print("path =", path)

        # ここで量子化モデルを再生成
        self.model = BitNetForMaskedLM(config)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

        # 再学習開始エポックを更新
        old_offset = self.epoch_offset
        loaded_epoch = int(os.path.splitext(os.path.basename(path))[0])
        self.epoch_offset = loaded_epoch + self.epoch_offset
        # すでに学習したエポックを除外/調整
        self.save_epochs = [x + (self.epoch_offset - old_offset) for x in self.save_epochs]

        print("Updated save_epochs:", self.save_epochs)
        print("Re-training will start from epoch =", self.epoch_offset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--vocab_size", default=100, type=str)
    parser.add_argument("--train_data_num", default=10000, type=int)
    parser.add_argument("--param_state", default='State', type=str)
    parser.add_argument("--thre_AD", default=0.05, type=float)
    params = vars(parser.parse_args())

    # Set up tokenizer and model configuration
    tokenizer_file = 'trained_tokenizer/vocab_size_{}/vocab.txt'.format(params["vocab_size"])
    tokenizer = BertWordPieceTokenizer(tokenizer_file)
    config = BertConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512
    )

    # Determine the device to use (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "saved_models/{}/saved_model_{}.pth".format(params["vocab_size"], params["train_data_num"])
    model = BertForMaskedLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"])

    # Set up datasets and data loaders
    train_data_path = "datasets/train/dataset_train_{}.txt".format(params["train_data_num"])
    train_dataset = MaskedTextDataset(train_data_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, collate_fn=DataHandler.collate_batch)

    test_data_path = "datasets/test/dataset_test_{}.txt".format(params["param_state"].lower())
    test_dataset = MaskedTextTestDataset(test_data_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    # Train and evaluate the model
    trainer = ModelTrainer(model, train_loader, optimizer, device, model_path, params["epochs"])
    trainer.train()

    tester = ModelTester(model, test_loader, params['use_proposed_method'], device)
    eval_results = tester.test(params["param_state"], params["thre_AD"], "results/miss_result_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    print("Evaluation Results:", eval_results)