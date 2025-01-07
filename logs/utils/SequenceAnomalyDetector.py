import re
import sys
from collections import Counter
from utils.ErrorHandling import readlines_log_file
import LogParser
from dataclasses import dataclass
from typing import List

@dataclass
class SequenceAnomalyResult:
    """
    シーケンス異常検知の結果を格納するクラス
    セッションとは、ログ行のContent部分の一連の順序を示す。

    Attributes:
        normal_session (NormalSession): 検出された正常なセッション
        anomaly_sessions (List[AnomalySession]): 検出された異常なセッション
    """
    @dataclass
    class NormalSession:
        """
        正常なセッションを格納するクラス

        Attributes:
            normal_session_content (List[str]): 検出された正常なセッション
        """
        normal_session_content: List[str]

    @dataclass
    class AnomalySession:
        """
        異常なセッションを格納するクラス

        Attributes:
            line_id (List[int]): 異常なセッションを構成するログ行番号
            anomaly_session_content (List[str]): 検出された異常なセッション
        """
        line_id: List[int]
        anomaly_session_content: List[str]

    normal_session: NormalSession
    anomaly_sessions: List[AnomalySession]


"""
Private Methods
"""
def mactch_df_list(df, target_list: list, column_name1: str, column_name2: str, column_name3: str):
    # LineIdに該当するComponentとContentを取得するリスト
    matching_ = []

    for target in target_list:
        # column_name1が存在するか確認
        match = df[df[column_name1] == target]
        if not match.empty:
            component = match[column_name2].values[0]
            content = match[column_name3].values[0]
            matching_.append(f'{component}: {content}')
        else:
            # column_name1が存在しない場合、Noneを追加
            print(f"Error: {column_name1} was not found in the dataframe.")
            # matching_.append(f"Error: {column_name1} was not found in the dataframe.")
            matching_.append("None")
    return matching_

def mactch_df_list_2column(df, target_list: list, column_name1: str, column_name2: str):
    # LineIdに該当するComponentとContentを取得するリスト
    matching_ = []

    for target in target_list:
        # column_name1が存在するか確認
        match = df[df[column_name1] == target]
        if not match.empty:
            col_2 = match[column_name2].values[0]
            matching_.append(col_2)
        else:
            # column_name1が存在しない場合、Noneを追加
            print(f"Error: {column_name1} was not found in the dataframe.")
            matching_.append("None")
    return matching_

def extract_logs_(df_structed, target_logs):
    logparser = LogParser.LogParser("")

    logCluL = logparser.parser.parse_raw_log(target_logs)
    template_str = "None"
    for logClust in logCluL:
        template_str = " ".join(logClust.logTemplate)

    matching_rows = df_structed[df_structed['EventTemplate'] == (template_str)]
    print(matching_rows)

    del logparser
    return matching_rows[["LineId","ParameterList"]], template_str

def extract_events_from_logs_(log_lines, component, session_length=5):
    """
    指定されたコンポーネントに関連する行からイベント名を抽出し、
    各セッション行番号を記録する。
    セッションの長さを任意に固定できるようにする。
    最後のセッションが指定した長さ未満の場合、無視する。
    """
    sessions = []
    current_session = []
    session_lines = []  # 各セッションの行番号を保持（各イベントの行番号を記録）

    for i, line in enumerate(log_lines):
        if component in line:
            # コンポーネント名の後に続くイベント名を抽出
            event = re.search(rf'{component}: (.+)', line)
            if event:
                event_name = event.group(1)
                if not current_session:
                    # 新しいセッションの開始
                    current_session = [event_name]
                    session_lines.append([i + 1])  # 開始行番号を記録
                else:
                    # 現在のセッションにイベントを追加
                    current_session.append(event_name)
                    session_lines[-1].append(i + 1)  # セッション内の行番号を追加

                # セッションの長さが指定した長さに達したらセッションを終了 :TODO Session(Window)で区切った余りを使用するかどうかの検討
                if len(current_session) == session_length:
                    sessions.append(current_session)
                    current_session = []  # セッションをリセット

    return sessions, session_lines


def extract_events_from_list_(data_list: list, session_length=5):
    """
    各セッションの開始行番号と終了行番号を記録する。
    セッションの長さを任意に固定できるようにする。
    最後のセッションが指定した長さ未満の場合、無視する。
    """
    sessions = []
    current_session = []
    session_lines = []  # 各セッションの開始と終了行番号を保持（開始, 終了）
    for i, event_name in enumerate(data_list):
        print(event_name)
        if event_name:
            if not current_session:
                # 新しいセッションの開始
                current_session = [event_name]
                session_lines.append(([i + 1]))
            else:
                # 現在のセッションにイベントを追加
                current_session.append(event_name)
                session_lines[-1].append(i + 1)  # セッション内の行番号を追加

            # セッションの長さが指定した長さに達したらセッションを終了
            if len(current_session) == session_length:
                sessions.append(current_session)
                current_session = []  # セッションをリセット

    return sessions, session_lines

def generate_all_rotations_(normal_order):
    """
    与えられた順序のすべての回転（循環シフト）を生成する。
    """
    rotations = []
    n = len(normal_order)
    for i in range(n):
        rotation = tuple(normal_order[i:] + normal_order[:i])
        rotations.append(rotation)
    return rotations

def detect_normal_order_(sessions):
    """
    全セッションからイベント順序を集計し、最も一般的な順序を「正常な順序」として返す。
    セッションが空の場合はエラーを出力する。
    """
    order_counts = Counter(tuple(session) for session in sessions)

    if not order_counts:
        raise ValueError("エラー: normal_orderが見つかりません. at SequenceAnomalyDetector: detect_normal_order_()")

    normal_order = order_counts.most_common(1)[0][0]  # 最も頻出する順序を取得
    return normal_order


def detect_anomaly_event_cycles(sessions, normal_order, session_lines):
    """
    正常なイベント順序から外れたセッションを検出し、ズレたイベントの行番号も取得する。
    循環シフトされた順序も正常とみなす。
    """
    anomalies = []
    anomaly_indexes = []
    shifted_event_lines = []

    # 正常な順序のすべての回転（循環シフト）を生成
    all_rotations = generate_all_rotations_(normal_order)

    for idx, session in enumerate(sessions):
        if tuple(session) not in all_rotations:
            anomalies.append(session)
            anomaly_indexes.append(idx)
            # セッション内のズレた行番号列を取得
            shifted_event_lines.append(session_lines[idx])

    return anomalies, shifted_event_lines


def detect_anomalies_grep_component(file_path, component_name, event_cycle):
    """
    :param file_path:
    :param component_name: コンポーネント名をユーザーが指定
    :param event_cycle: セッションの長さを固定（例: 5イベントごとに分割）
    :return:
    """

    # ログファイルの読み込み
    log_data = readlines_log_file(file_path)

    # ログからイベントを抽出し、行番号を記録
    sessions, session_lines = extract_events_from_logs_(log_data, component_name, session_length=event_cycle)

    # 正常な順序の自動検出
    normal_order = detect_normal_order_(sessions)

    # 異常セッションの検出
    anomalies, shifted_event_lines = detect_anomaly_event_cycles(sessions, normal_order, session_lines)

    anomaly_sessions_ = []
    for idx, (anomaly, shifted_lines) in enumerate(zip(anomalies, shifted_event_lines), 1):
        anomaly_sessions_.append(SequenceAnomalyResult.AnomalySession(line_id=shifted_lines,
                                                                     anomaly_session_content=anomaly))

    return SequenceAnomalyResult(
        normal_session=SequenceAnomalyResult.NormalSession(normal_session_content=list(normal_order)),
        anomaly_sessions=anomaly_sessions_)

def detect_anomalies_grep_component_and_template(file_path, component_name, event_cycle, target_logs, input_regex):
    """
    :param file_path:
    :param component_name:
    :param event_cycle:
    :param target_logs: 確認したいシーケンスに使用するログ（テンプレート）がすべて含まれた配列
    :param input_regex: テンプレートを抽出するのに必要な正規表現
    :return:
    用途：一つのComponent内に複数のTemplateがあり、指定したTemplateのみのシーケンスパターンを見たい場合
    """

    # ログファイルの読み込み
    df_structed, df_templates, logparser = LogParser.parse_log_file_(file_path, regex_list=input_regex)
    # Parse Target Log
    template_strs = LogParser.parse_logs(logparser, target_logs)
    # Extract dataframe
    df_structed = df_structed[df_structed['Component'] == component_name]
    df_structed = df_structed[df_structed['EventTemplate'].isin(template_strs)]

    line_ids = df_structed["LineId"].tolist()
    content_list = df_structed["Content"].tolist()

    # ログからイベントを抽出し、行番号を記録
    sessions, session_lines = extract_events_from_list_(content_list, session_length=event_cycle)

    # 正常な順序の自動検出
    normal_order = detect_normal_order_(sessions)

    # 異常セッションの検出
    anomalies, shifted_event_lines = detect_anomaly_event_cycles(sessions, normal_order, session_lines)

    # Replace numbers in shifted_event_lines with corresponding values from line_ids
    shifted_event_lines = [[line_ids[num - 1] for num in sublist] for sublist in shifted_event_lines]

    anomaly_sessions_ = []
    for idx, (anomaly, shifted_lines) in enumerate(zip(anomalies, shifted_event_lines), 1):
        anomaly_sessions_.append(SequenceAnomalyResult.AnomalySession(line_id=shifted_lines,
                                                                      anomaly_session_content=anomaly))

    return SequenceAnomalyResult(
        normal_session=SequenceAnomalyResult.NormalSession(normal_session_content=list(normal_order)),
        anomaly_sessions=anomaly_sessions_)


def detect_anomalies_target_logs(file_path, target_sequences, target_logs, input_regex):
    """
    :param file_path: ログファイルのパス
    :param target_sequences: 確認したいログの正常な順序
    :param target_logs:
    :param input_regex: テンプレートを抽出するのに必要な正規表現
    :return:
    用途：
    """
    event_cycle = len(target_sequences)
    component_list = []
    normal_order = []
    # Regex is Android Format.
    regex = re.compile('^(?P<Date>.*?)\\s+(?P<Time>.*?)\\s+(?P<Pid>.*?)\\s+(?P<Tid>.*?)\\s+(?P<Level>.*?)\\s+(?P<Component>.*?):\\s+(?P<Content>.*?)$')
    headers = ['Date', 'Time', 'Pid', 'Tid', 'Level', 'Component', 'Content']
    for line in target_sequences:
        match = regex.search(line.strip())
        message = [match.group(header) for header in headers]
        component_list.append(message[5])
        normal_order.append(message[6])
    component_list = set(component_list)
    normal_order = tuple(normal_order)

    # ログファイルの読み込み
    df_structed, df_templates, logparser = LogParser.parse_log_file_(file_path, regex_list=input_regex)
    # Parse Target Log
    template_strs = LogParser.parse_logs(logparser, target_sequences+target_logs)
    # Extract dataframe
    df_structed = df_structed[df_structed['Component'].isin(component_list)]
    df_structed = df_structed[df_structed['EventTemplate'].isin(template_strs)]

    line_ids = df_structed["LineId"].tolist()
    content_list = df_structed["Content"].tolist()

    # ログからイベントを抽出し、行番号を記録
    sessions, session_lines = extract_events_from_list_(content_list, session_length=event_cycle)

    # 正常な順序の自動検出
    # normal_order = detect_normal_order_(sessions)

    # 異常セッションの検出
    anomalies, shifted_event_lines = detect_anomaly_event_cycles(sessions, normal_order, session_lines)

    # Replace numbers in shifted_event_lines with corresponding values from line_ids
    shifted_event_lines = [[line_ids[num - 1] for num in sublist] for sublist in shifted_event_lines]

    anomaly_sessions_ = []
    for idx, (anomaly, shifted_lines) in enumerate(zip(anomalies, shifted_event_lines), 1):
        anomaly_sessions_.append(SequenceAnomalyResult.AnomalySession(line_id=shifted_lines,
                                                                      anomaly_session_content=anomaly))

    return SequenceAnomalyResult(
        normal_session=SequenceAnomalyResult.NormalSession(normal_session_content=list(normal_order)),
        anomaly_sessions=anomaly_sessions_)

def detect_anomalies(file_path: str, target_sequences: list, input_regex: list):
    """
    :param file_path: ログファイルのパス
    :param target_sequences: 確認したいログの正常な順序
    :param input_regex: テンプレートを抽出するのに必要な正規表現
    :return: SequenceAnomalyResult
    機能：指定したログ順序から逸脱する順序を見つける
    条件：見つけたいログ順序が発見できるような完全なRegexが必要
    """
    event_cycle = len(target_sequences)
    component_list = []
    normal_order = []
    # Regex is Android Format.
    regex = re.compile('^(?P<Date>.*?)\\s+(?P<Time>.*?)\\s+(?P<Pid>.*?)\\s+(?P<Tid>.*?)\\s+(?P<Level>.*?)\\s+(?P<Component>.*?):\\s+(?P<Content>.*?)$')
    headers = ['Date', 'Time', 'Pid', 'Tid', 'Level', 'Component', 'Content']
    for line in target_sequences:
        match = regex.search(line.strip())
        message = [match.group(header) for header in headers]
        component_list.append(message[5])
        normal_order.append(message[6])
    component_list = set(component_list)

    # ログファイルの読み込み
    df_structed, df_templates, logparser = LogParser.parse_log_file_(file_path, regex_list=input_regex)

    # Parse Target Sequences
    normal_order = LogParser.parse_logs(logparser, target_sequences)
    print("正常な順序のTemplate: ", normal_order)
    normal_order_templates = normal_order

    template_strs = set(normal_order)

    # DataFrameからlist内のContentに該当するEventIdを取得
    # 元のnormal_orderの順序性を保ちながらUniqueEventIdを取得
    normal_order = [df_templates[df_templates['UniqueEventTemplate'] == template]['UniqueEventId'].values[0]
                          for template in normal_order if template in df_templates['UniqueEventTemplate'].values]
    if len(normal_order) < event_cycle:
        raise ValueError("エラー: 正常な順序のログを上手くテンプレート化できませんでした。 at SequenceAnomalyDetector: detect_anomalies()")

    del logparser, df_templates
    # Extract dataframe
    df_structed = df_structed[df_structed['Component'].isin(component_list)]
    df_structed = df_structed[df_structed['EventTemplate'].isin(template_strs)]

    line_ids = df_structed["LineId"].tolist()
    event_id_list = df_structed["EventId"].tolist()

    # ログからイベントを抽出し、行番号を記録
    sessions, session_lines = extract_events_from_list_(event_id_list, session_length=event_cycle)

    # 異常セッションの検出
    anomalies, shifted_event_lines = detect_anomaly_event_cycles(sessions, normal_order, session_lines)

    # Replace numbers in shifted_event_lines with corresponding values from line_ids
    shifted_event_lines = [[line_ids[num - 1] for num in sublist] for sublist in shifted_event_lines]


    anomaly_sessions_ = []
    for idx, (anomaly, shifted_lines) in enumerate(zip(anomalies, shifted_event_lines), 1):
        # EventId -> Content
        anomaly = mactch_df_list(df_structed, shifted_lines, column_name1="LineId", column_name2="Component", column_name3="Content")
        anomaly_sessions_.append(SequenceAnomalyResult.AnomalySession(line_id=shifted_lines,
                                                                      anomaly_session_content=anomaly))
    return SequenceAnomalyResult(
        normal_session=SequenceAnomalyResult.NormalSession(normal_session_content=normal_order_templates),
        anomaly_sessions=anomaly_sessions_)


def detect_anomalies_content_cycle(file_path: str, target_sequences: list, input_regex: list):
    """
    :param file_path: ログファイルのパス
    :param target_sequences: 確認したいログの正常な順序
    :param input_regex: テンプレートを抽出するのに必要な正規表現
    :return: SequenceAnomalyResult
    用途：detect_anomalies()で正確な正規表現が得られない時.パラメータ情報が丸め込まれた結果、検知できない時。
    システムフロー
    1. 正常順序のテンプレートを作成
    2. 抽出したログテンプレートでログファイル（DataFrame）をgrep
    3. grepしたDataFrameのContent部分(Template + Parameter)を用いて異常検知を行う。
    """
    event_cycle = len(target_sequences)
    component_list = []
    normal_order = []
    # Regex is Android Format.
    regex = re.compile('^(?P<Date>.*?)\\s+(?P<Time>.*?)\\s+(?P<Pid>.*?)\\s+(?P<Tid>.*?)\\s+(?P<Level>.*?)\\s+(?P<Component>.*?):\\s+(?P<Content>.*?)$')
    headers = ['Date', 'Time', 'Pid', 'Tid', 'Level', 'Component', 'Content']
    for line in target_sequences:
        match = regex.search(line.strip())
        message = [match.group(header) for header in headers]
        component_list.append(message[5])
        normal_order.append(message[6])
    component_list = set(component_list)

    # ログファイルの読み込み
    df_structed, df_templates, logparser = LogParser.parse_log_file_(file_path, regex_list=input_regex)

    # Parse Target Sequences
    template_strs = set(LogParser.parse_logs(logparser, target_sequences))
    del logparser, df_templates

    # Extract dataframe
    df_structed = df_structed[df_structed['Component'].isin(component_list)]
    df_structed = df_structed[df_structed['EventTemplate'].isin(template_strs)]

    line_ids = df_structed["LineId"].tolist()
    content_list = df_structed["Content"].tolist()

    # ログからイベントを抽出し、行番号を記録
    sessions, session_lines = extract_events_from_list_(content_list, session_length=event_cycle)

    # 異常セッションの検出
    anomalies, shifted_event_lines = detect_anomaly_event_cycles(sessions, normal_order, session_lines)

    # Replace numbers in shifted_event_lines with corresponding values from line_ids
    shifted_event_lines = [[line_ids[num - 1] for num in sublist] for sublist in shifted_event_lines]

    anomaly_sessions_ = []
    for idx, (anomaly, shifted_lines) in enumerate(zip(anomalies, shifted_event_lines), 1):
        # EventId -> Content
        anomaly_sessions_.append(SequenceAnomalyResult.AnomalySession(line_id=shifted_lines,
                                                                      anomaly_session_content=anomaly))
    return SequenceAnomalyResult(
        normal_session=SequenceAnomalyResult.NormalSession(normal_session_content=normal_order),
        anomaly_sessions=anomaly_sessions_)

def detect_template_anomalies_specified_component(file_path: str, component_name: str, event_cycle: int, input_regex: list=[]):
    """
    :param file_path: ログファイルのパス
    :param component_name: 確認したいログのComponent名
    :param input_regex: テンプレートを抽出するのに必要な正規表現
    :return: SequenceAnomalyResult
    """

    # ログファイルの読み込み
    df_structed, df_templates, logparser = LogParser.parse_log_file_(file_path, input_regex)
    del df_templates, logparser

    # Extract dataframe
    df_structed = df_structed[df_structed['Component'] == component_name]
    template_data = df_structed["EventId"].tolist()
    line_ids = df_structed["LineId"].tolist()

    # ログからイベントを抽出し、行番号を記録
    sessions, session_lines = extract_events_from_list_(template_data, session_length=event_cycle)

    # 正常な順序の自動検出
    normal_order = detect_normal_order_(sessions)

    # 異常セッションの検出
    anomalies, shifted_event_lines = detect_anomaly_event_cycles(sessions, normal_order, session_lines)

    normal_order = list(normal_order)

    normal_order_templates = mactch_df_list_2column(df_structed, normal_order, column_name1="EventId",
                                                    column_name2="EventTemplate")

    # Replace numbers in shifted_event_lines with corresponding values from line_ids
    shifted_event_lines = [[line_ids[num - 1] for num in sublist] for sublist in shifted_event_lines]

    anomaly_sessions_ = []
    for idx, (anomaly, shifted_lines) in enumerate(zip(anomalies, shifted_event_lines), 1):
        # EventId -> Content
        anomaly = mactch_df_list(df_structed, shifted_lines, column_name1="LineId", column_name2="Component",
                                 column_name3="Content")
        anomaly_sessions_.append(SequenceAnomalyResult.AnomalySession(line_id=shifted_lines,
                                                                      anomaly_session_content=anomaly))
    return SequenceAnomalyResult(
        normal_session=SequenceAnomalyResult.NormalSession(normal_session_content=normal_order_templates),
        anomaly_sessions=anomaly_sessions_)

def detect_template_anomalies(file_path, event_cycle):
    """
    :param file_path:
    :param component_name: コンポーネント名をユーザーが指定
    :param event_cycle: セッションの長さを固定（例: 5イベントごとに分割）
    :return:
    """

    # ログファイルの読み込み
    df_structed, _ = LogParser.parse_log_file(file_path)
    template_data = df_structed["EventId"].tolist()

    # ログからイベントを抽出し、行番号を記録
    sessions, session_lines = extract_events_from_list_(template_data, session_length=event_cycle)

    # 正常な順序の自動検出
    normal_order = detect_normal_order_(sessions)

    # 異常セッションの検出
    anomalies, shifted_event_lines = detect_anomaly_event_cycles(sessions, normal_order, session_lines)

    anomaly_sessions_ = []
    for idx, (anomaly, shifted_lines) in enumerate(zip(anomalies, shifted_event_lines), 1):
        anomaly_sessions_.append(SequenceAnomalyResult.AnomalySession(line_id=shifted_lines,
                                                                      anomaly_session_content=anomaly))

    return SequenceAnomalyResult(
        normal_session=SequenceAnomalyResult.NormalSession(normal_session_content=list(normal_order)),
        anomaly_sessions=anomaly_sessions_)
