import sys

import numpy as np
# from logparser import Drain
import os
import time

def detect_anomalies_with_tf(tf_list: list, event_id_list: list, anomaly_mode: str ="IQR"):
    """
    :param tf_list: tf-valueのリスト
    :param event_id_list: event IDのリスト (tf_listと同じ長さで、各要素に対応)
    :param anomaly_mode: 異常検出モード ("IQR", "Z-Score", "Threshold_base")
    :return: [dict1, dict2, ...]
    dict = {"EventId": event_id,
            "Score": score,
            "Label": label }
    """
    print("start detect_anomalies_with_tf()")
    mode_candidates = ["IQR", "Z-Score", "Threshold_base"]
    anomaly_mode = mode_candidates[1]
    anomalies = []
    anomaly_event_ids = []
    labels = []
    scores = []
    start = time.perf_counter()  # 計測開始

    tf_array = np.array(tf_list)

    if anomaly_mode == "IQR":
        # Q1（第1四分位数）とQ3（第3四分位数）を計算
        Q1 = np.percentile(tf_array, 25)
        Q3 = np.percentile(tf_array, 75)

        # IQR（四分位範囲）を計算
        IQR = Q3 - Q1

        # 異常の閾値を計算
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 異常を特定
        # for i, x in enumerate(tf_list):
        #     if x < lower_bound or x > upper_bound:
        #         anomalies.append(x)
        #         anomaly_event_ids.append(event_id_list[i])
        # print(f"IQR異常値: {anomalies}")
        # print(f"IQR異常EventId: {anomaly_event_ids}")
        labels = [(x < lower_bound or x > upper_bound) for x in tf_list]
        scores = [None for i in range(len(tf_list))]

    elif anomaly_mode == "Z-Score":
        # tf値に対するzスコアを計算
        tf_mean = np.mean(tf_array)
        tf_std = np.std(tf_array)
        z_scores = (tf_array - tf_mean) / tf_std

        # zスコアが一定の閾値を超えた場合に異常としてフラグを立てる
        # for i, z in enumerate(z_scores):
        #     if abs(z) > 3:
        #         anomalies.append(tf_list[i])
        #         anomaly_event_ids.append(event_id_list[i])
        #         labels.append()
        # print(f"Z-Score異常値: {anomalies}")
        # print(f"Z-Score異常EventId: {anomaly_event_ids}")
        scores = z_scores
        labels = [z_score > 2 for z_score in z_scores ]

    elif anomaly_mode == "Threshold_base":
        # 閾値を設定 (例えば、tf値が0.001以下を異常とみなす)
        threshold = 0.001

        # 異常を特定 (tf値が閾値よりも低い場合)
        # for i, x in enumerate(tf_list):
        #     if x < threshold:
        #         anomalies.append(x)
        #         anomaly_event_ids.append(event_id_list[i])
        # print(f"Threshold異常値: {anomalies}")
        # print(f"Threshold異常EventId: {anomaly_event_ids}")
        for i, x in enumerate(tf_list):
            scores.append(None)
            labels.append(x < threshold)

    end = time.perf_counter()  # 計測終了
    print('Calculated Anomaly Time is {:.2f}'.format((end - start) / 60))

    output_info = []

    # zip() で3つの配列を同時にループし、Dictに格納
    for event_id, score, label in zip(event_id_list, scores, labels):
        # 各要素を辞書に格納
        entry = {
            "EventId": event_id,
            "Score": score,
            "Label": label
        }
        # 配列に追加
        output_info.append(entry)

    print("finish detect_anomalies_with_tf()")
    return output_info

# Usage example
if __name__ == "__main__":
    """
    input: file_path
    output: DataFrame
    """
    # datasets_dir="datasets"
    # input_file_path = "datasets/test_logcat_15k.log"
    # input_file_path = "datasets/test.test"
    # input_file_path = "datasets/no_read_permission_file.log"

    # df_structed, df_templates = preprocess_template_anomaly_detection(input_file_path)
    # print(df_structed)
    # print()
    # print("--------")
    # detect_anomalies_with_tf(df_templates["tf"].tolist(), df_templates["EventId"].tolist())
    # sys.exit()
