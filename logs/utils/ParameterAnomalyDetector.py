import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from collections import Counter
import LogParser

from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import DistilBertTokenizer

"""
Parameterを用いた異常検知システム
"""

"""
Private Methods
"""
def extract_logs_(df_structed, target_logs):
    logparser = LogParser.LogParser("")

    logCluL = logparser.parser.parse_raw_log(target_logs)
    template_str = "None"
    for logClust in logCluL:
        template_str = " ".join(logClust.logTemplate)

    matching_rows = df_structed[df_structed['EventTemplate'] == (template_str)]

    del logparser
    return matching_rows[["LineId","ParameterList"]], template_str

def create_param_dict_(parameter_list):
    # 縦列のデータを取り出し、それぞれの列を辞書に保存
    param_dict = {}

    # データを縦方向で分割して取得
    for i, col in enumerate(zip(*parameter_list)):
        key = f'Param{i+1}'  # Keyの部分をParam1, Param2, ... とする
        param_dict[key] = list(col)  # 各縦列のデータをリストに変換して格納

    return param_dict

def is_numeric_string_(s):
    """
    与えられた文字列が数値を表すかどうかを判定する関数。

    :param s: 判定する文字列
    :return: 数値を表す場合はTrue、そうでない場合はFalse
    """

    try:
        float(s)  # 文字列をfloatに変換してみる
        return True
    except ValueError:
        # 変換できない場合は数値ではない
        return False

####
# Methods of Calculating Anomaly Score
####
### For Numeric
class AnomalyDetectionLibrary:
    def __init__(self):
        # 異常検知に使う関数を辞書に登録
        self.methods_numeric = {
            'IQR': self.iqr,
            'Z-Score': self.zscore,
            'K_NN': self.k_nn,
            'DBSCAN': self.dbscan,
            'Occurrence': self.occurrence,
            'Custom': self.custom_anomaly_detection
        }
        self.methods_string = {
            'Sentiment': self.sentiment_analysis,
            'Occurrence': self.occurrence,
            'Custom': self.custom_anomaly_detection
        }

    def execute_detection(self, data, is_numeric, method_name_numeric, method_name_string="Sentiment"):
        """
        データと異常検知のメソッド名を受け取り、指定された異常検知方法を実行する。
        :param data: 入力データ
        :param is_numeric: NemericかどうかのBool値
        :param method_name_numeric: Numeric型の異常検知関数を指定（'iqr', 'zscore', 'custom' など）
        :param method_name_string: String型の異常検知関数を指定（'Sentiment', 'custom' など）
        :return: 異常検知結果の情報(scores, thresholds labels)
            dict={'LineId': 416, “Parameter”: 1250, 'Score': None, ‘Threshold’: [250, 400], 'Label' True}
        """
        # mode_candidates = ["IQR", "Z-Score", "K_NN", "DBSCAN", "Occurrence"]

        if is_numeric:
            # method_name_numeric=mode_candidates[4]
            methods = self.methods_numeric
            method_name = method_name_numeric
        else:
            methods = self.methods_string
            method_name = method_name_string
        if method_name in methods:
            detection_method = methods[method_name]
            return detection_method(data)
        else:
            raise ValueError(f"指定されたメソッド '{method_name}' は存在しません。")

    ####
    # For Numeric Parameter
    ####
    def iqr(self, data):
        """
        IQR (四分位範囲) を使った異常検知の例
        """

        # 2. 四分位数範囲（IQR）
        q1 = np.percentile(data, 25).item()
        q3 = np.percentile(data, 75).item()
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        labels = [(x < lower_bound or x > upper_bound) for x in data]
        scores = [None for i in range(len(data))]
        return scores, f"parameter < {lower_bound} or parameter > {upper_bound}", labels


    def zscore(self, data):
        """
        Zスコアを使った異常検知の例
        """
        # 1. Z-スコア（標準スコア）
        z_scores = np.abs(stats.zscore(data))
        labels = [z_score > 2 for z_score in z_scores ]
        return z_scores, "score > 2", labels

    def k_nn(self, data):
        data = np.array(data).reshape(-1, 1)
        # 3. 距離ベースの手法（k-最近傍法）
        nbrs = NearestNeighbors(n_neighbors=2).fit(data)
        distances, indices = nbrs.kneighbors(data)
        distances = distances[:, 1]
        threshold = 2 * np.mean(distances)
        labels = [distance > threshold for distance in distances]
        return distances.tolist(), f"score > {threshold}", labels

    def dbscan(self, data):
        data = np.array(data).reshape(-1, 1)
        # 4. 密度ベースの手法（DBSCAN）
        dbscan = DBSCAN(eps=3, min_samples=2).fit(data)
        outliers_dbscan = np.where(dbscan.labels_ == -1)
        labels = [label == -1 for label in dbscan.labels_]
        return dbscan.labels_, f"score == -1", labels

    def occurrence(self, data):
        # print(data)
        # data = [12025.0, 5784.0, 23484.0, 5769.0, 12236.0, 12236.0]

        # 各パラメータの出現回数を計算
        occurrences = Counter(data)
        occurrences_values = list(occurrences.values())

        # anomaly_mode = "IQR"
        # if anomaly_mode == "IQR":
        #     # 2. 四分位数範囲（IQR）
        #     q1 = np.percentile(occurrences_values, 25).item()
        #     q3 = np.percentile(occurrences_values, 75).item()
        #     iqr = q3 - q1
        #     lower_bound = q1 - 1.5 * iqr
        #     upper_bound = q3 + 1.5 * iqr
        #     labels = []
        #     scores = []
        #     for x in data:
        #         occurrence = occurrences[x]
        #         labels.append((occurrence < lower_bound or occurrence > upper_bound))
        #         scores.append(occurrence)
        #     print(lower_bound, upper_bound)
        #     sys.exit()
        #
        #     return scores, f"parameter < {lower_bound} or parameter > {upper_bound}", labels
        # if anomaly_mode == "DBSCAN":
            # data_ = np.array(occurrences_values).reshape(-1, 1)
            # # 4. 密度ベースの手法（DBSCAN）
            # dbscan = DBSCAN(eps=3, min_samples=2).fit(data_)
            # dict_ = {}
            # for i, occurrence in enumerate(occurrences_values):
            #     dict_[str(occurrence)] = dbscan.labels_[i]
            #
            # labels = []
            # scores = []
            # for parameter in occurrences_values:
            #     labels.append(dict_[str(occurrence)])
            #     scores.append(parameter)
            # outliers_dbscan = np.where(dbscan.labels_ == -1)
            # labels = [label == -1 for label in dbscan.labels_]
            # return dbscan.labels_, f"score == -1", labels


        # 3. 距離ベースの手法（k-最近傍法）
        data_ = np.array(occurrences_values).reshape(-1, 1)
        nbrs = NearestNeighbors(n_neighbors=2).fit(data_)
        distances, indices = nbrs.kneighbors(data_)
        distances = distances[:, 1]
        threshold = 2 * np.mean(distances)

        dict_ = {}
        for i, occurrence_ in enumerate(occurrences_values):
            dict_[str(occurrence_)] = distances[i]

        labels = []
        scores = []
        for parameter in data:
            occurrence_ = occurrences[parameter]
            score = dict_[str(occurrence_)]
            labels.append(score > threshold)
            scores.append(score)

        return scores, f"score > {threshold}", labels

    ####
    # For String Parameter
    ####
    def sentiment_analysis(self, data):
        """
        License: ページ内にLicenseの文字があります。
        'distilbert-base-uncased-finetuned-sst-2-english'=License: Apache-2.0: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
        'distilbert-base-uncased'=License: Apache-2.0: https://huggingface.co/distilbert/distilbert-base-uncased
        "sentiment-analysis": 「transformers」の一部であり、Apache License 2.0(transformers自体のLicense)
        - 調査した内容をDocumentフォルダに格納（Document/Transformer_Model_Licenses_JP.txt）

        Warning:
        transformers, pytorchを入れた後互換性の問題でnumpy==1.26.4を入れた。
        似た問題が発生する可能性あり。
        Error文:
         "venv_intern/lib/python3.9/site-packages/transformers/pipelines/text_classification.py",
          line 208, in postprocess
            outputs = outputs.float().numpy()
            RuntimeError: Numpy is not available
        """
        # 指定したフォルダにモデルとトークナイザをダウンロード
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english', cache_dir='pretrained_models')
        tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased', cache_dir='pretrained_models')
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        scores = []
        labels = []
        for x in data:
            predicted_label, score = classifier(x)[0].items()
            scores.append(score[1])
            labels.append(predicted_label[1] == "NEGATIVE")
        return scores, "score <= 0.5", labels

    def custom_anomaly_detection(self, data):
        """
        カスタムの異常検知アルゴリズム
        """
        # 独自の異常検知ロジックをここに実装
        pass

def create_output_info(line_ids, parameters, scores, thresholds, labels):
    # 各要素を格納する配列
    output_info = []

    # zip() で3つの配列を同時にループし、Dictに格納
    for line_id, param, score, label in zip(line_ids, parameters, scores, labels):
        # 各要素を辞書に格納
        entry = {
            "LineId": line_id,
            "Parameter": param,
            "Score": score,
            "Threshold": thresholds,
            "Label": label
        }
        # 配列に追加
        output_info.append(entry)
    return output_info

"""
Public Methods (Called by LLM)
"""
def detect_anomalies_with_parse(file_path:str, target_log:str =None, anomaly_mode_numeric: str ="IQR", anomaly_mode_string: str = "Sentiment",  regex_list: list=[]):
    print(regex_list)
    df_structed, df_templates, logparser = LogParser.parse_log_file_(file_path, regex_list=regex_list)
    # Parse Target Log
    template_strs = LogParser.parse_logs(logparser, [target_log])
    # Extract dataframe
    matching_rows = df_structed[df_structed['EventTemplate'] == (template_strs[0])]

    del logparser
    return detect_anomalies(matching_rows, anomaly_mode_numeric=anomaly_mode_numeric, anomaly_mode_string=anomaly_mode_string)

def detect_anomalies(dataframe, target_log: str=None, anomaly_mode_numeric: str ="IQR", anomaly_mode_string: str = "Sentiment"):
    """
    :param dataframe:
    :param target_log:
    :param anomaly_mode:
    :return: output_dict
    """
    if target_log:
        # 抽出する必要がある場合
        dataframe, event_template = extract_logs_(dataframe, [target_log])
    # 抽出したDataFrameがある場合

    if dataframe.empty:
        raise Exception("Dataframe is empty.")
        return None

    line_id_list = dataframe["LineId"].tolist()
    parameter_list = dataframe["ParameterList"].tolist()

    if not parameter_list[0]:
        raise Exception("Parameter is empty.")
        return None

    param_dict = create_param_dict_(parameter_list)

    anomaly_lib = AnomalyDetectionLibrary()
    output_dict = {}
    param_num = 1
    # Start Anomaly Detection
    for key, parameters in param_dict.items():
        dict_key = "Param"+str(param_num)
        if is_numeric_string_(parameters[0]):
            print("{} is numeric_string".format(dict_key))
            parameters = [float(v) for v in parameters]
            scores, thresholds, labels = anomaly_lib.execute_detection(parameters, True, anomaly_mode_numeric, anomaly_mode_string)
        else:
            print("{} is string".format(dict_key))
            scores, thresholds, labels = anomaly_lib.execute_detection(parameters, False, anomaly_mode_numeric, anomaly_mode_string)
        output_info = create_output_info(line_id_list, parameters, scores, thresholds, labels)
        output_dict[dict_key] = output_info
        param_num += 1

    # print("output_dict: ", output_dict)
    return output_dict

