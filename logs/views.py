from django.shortcuts import render
from .forms import LogFileUploadForm  # ここを追加
# アップロードファイルの保存ディレクトリを指定
import os
from .utils import LogParser, TemplaterAnomalyDetector
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from django.http import HttpResponse

UPLOAD_DIR = "media/uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def index(request):
    """トップページ"""
    return render(request, 'logs/index.html')

def upload_log(request):
    content = None
    results_ = {'html_table': None,
            'event_graph_url': None,
            'component_graph_url': None,
            'top_event_ids': None,
            'top_components': None,
            'total_event_ids': None,
            'total_components': None
            }

    if request.method == 'POST':
        form = LogFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            # アップロードされたファイルを保存
            with open(file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            if 'parse' in request.POST:  # Parse ボタンが押された場合
                try:
                    print("Parse button clicked")
                    # Parse処理を実行 (仮の処理としてファイル名変更を実装)
                    csv_file_path = file_path.replace('.log', '.csv').replace('.txt', '.csv')
                    with open(csv_file_path, 'w') as f:
                        f.write("This is a parsed CSV file.\n")  # 仮の内容を追加
                    # event_graph_url, component_graph_url, html_table = template(file_path)
                    results_ = template(file_path)
                    # html_table = parse_result_view()
                except Exception as e:
                    print(f"Error during parsing: {e}")
                    return render(request, 'logs/upload_log.html', {'error': str(e)})
            else:  # 通常アップロード処理
                with open(file_path, 'r') as f:
                    content = f.read()

    else:
        form = LogFileUploadForm()

    return render(
        request,
        'logs/upload_log.html',
        {
            'form': form,
            'content': content,
            'html_table': results_["html_table"],
            'event_graph_url': results_["event_graph_url"],
            'component_graph_url': results_["component_graph_url"],
            'top_event_ids': results_["top_event_ids"],
            'top_components': results_["top_components"],
            'total_event_ids': results_["total_event_ids"],
            'total_components': results_["total_components"]
        }
    )

def future_app1(request):
    """将来のアプリケーション1ページ"""
    return render(request, 'logs/future_app1.html')

def future_app2(request):
    """将来のアプリケーション2ページ"""
    return render(request, 'logs/future_app2.html')

def about(request):
    """Aboutページ"""
    return render(request, 'logs/about.html')

def template(input_file_path):

    print("Start template(input_file_path)")
    df_structed, df_templates = LogParser.parse_log_file(input_file_path)
    print(df_structed)
    print("----------"*2)
    print(df_templates)
    # output_info = TemplaterAnomalyDetector.detect_anomalies_with_tf(df_templates["tf"].tolist(), df_templates["UniqueEventId"].tolist())
    # print(output_info)

    # EventIdごとの出現頻度を計算
    event_counts = df_structed['EventId'].value_counts()
    total_event_ids = len(event_counts)
    top_event_ids = event_counts.head(5)

    # グラフの作成
    plt.figure(figsize=(8, 4))
    event_counts.plot(kind='bar', color='skyblue')
    plt.title("EventId Frequency")
    plt.xlabel("EventId")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # グラフを画像として保存
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    event_graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    # Componentごとの出現頻度を計算
    component_counts = df_structed['Component'].value_counts()
    total_components = len(component_counts)
    top_components = component_counts.head(5)

    # Componentのグラフ作成
    plt.figure(figsize=(8, 4))
    component_counts.plot(kind='bar', color='coral')
    plt.title("Component Frequency", fontsize=16)
    plt.xlabel("Component", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # グラフを画像として保存
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    component_graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    # HTMLテーブルに変換 (DataTables用のクラスを追加)
    html_table = df_structed.to_html(index=False, escape=False, header=True)

    # return graph_url, component_graph_url, html_table
    return {'html_table': html_table,
            'event_graph_url': event_graph_url,
            'component_graph_url': component_graph_url,
            'top_event_ids': top_event_ids,
            'top_components': top_components,
            'total_event_ids': total_event_ids,
            'total_components': total_components
            }
def parse_result_view():
    # サンプルデータ
    data = {
        "LineId": [1, 2],
        "Date": ["03-17", "03-17"],
        "Time": ["16:13:38.811", "16:13:38.819"],
        "Pid": [1702, 1702],
        "Tid": [2395, 8671],
        "Level": ["D", "D"],
        "Component": ["WindowManager", "PowerManagerService"],
        "Content": [
            "printFreezingDisplayLogsopening app wtoken = AppWindowToken{9f4ef63 token=Token{a64f992 ActivityRecord{de9231d u0 com.tencent.qt.qtl/.activity.info.NewsDetailXmlActivity t761}}}, allDrawn= false, startingDisplayed =  false, startingMoved =  false, isRelaunching =  false",
            'acquire lock=233570404, flags=0x1, tag="View Lock", name=com.android.systemui, ws=null, uid=10037, pid=2227'
        ],
        "EventId": ["3caee421", "60843176"],
        "EventTemplate": [
            "printFreezingDisplayLogsopening app wtoken = AppWindowToken{<*> token=Token{<*> ActivityRecord{<*> u0 <*>/.<*> t761}}}, allDrawn= false, startingDisplayed = false, startingMoved = false, isRelaunching = false",
            'acquire lock=<*>, flags=<*>, tag="View Lock", name=<*>, ws=null, uid=<*>, pid=<*>'
        ],
        "ParameterList": [
            ['9f4ef63', 'a64f992', 'de9231d', 'com.tencent.qt.qtl', 'activity.info.NewsDetailXmlActivity'],
            ['233570404', '0x1', 'com.android.systemui', '10037', '2227']
        ]
    }

    # DataFrameを生成
    df = pd.DataFrame(data)

    # HTMLテーブルに変換 (DataTables用のクラスを追加)
    html_table = df.to_html(index=False, escape=False, header=True)

    return html_table

####################
# Anomaly Detection
####################
# 新しいAnomaly Detection用のアップロードページ
def anomaly_detection_upload(request):
    if request.method == 'POST':
        if 'upload' in request.POST:
            # ログファイルのアップロード処理
            form = LogFileUploadForm(request.POST, request.FILES)
            if form.is_valid():
                uploaded_file = request.FILES['file']
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

                # アップロードされたファイルを保存
                with open(file_path, 'wb') as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)

            return HttpResponse("File uploaded successfully.")

        elif 'parse' in request.POST:
            # Parse処理
            return HttpResponse("Parse process completed.")

        elif 'train' in request.POST:
            # モデル学習処理
            model()  # 学習用の空の関数を呼び出し
            return HttpResponse("Training process completed.")

        elif 'test' in request.POST:
            # テスト処理
            return HttpResponse("Testing process completed.")

    return render(request, 'anomaly_detection/upload_log.html')
def anomaly_detection(request):
    return render(request, 'anomaly_detection/anomaly_detection.html', {
        'title': 'Anomaly Detection [Beta]',
        'description': 'This is the beta version of our anomaly detection feature. You can explore its capabilities here.',
    })

# 空のモデル関数
def model():
    pass

# 学習処理
def train(request):
    if request.method == 'POST':
        model()  # モデル学習関数の呼び出し
        return HttpResponse("Training process completed.")
    return render(request, 'log_ad_app/train.html')

# テスト処理
def test(request):
    if request.method == 'POST':
        # テスト用ロジックをここに追加
        return HttpResponse("Testing process completed.")
    return render(request, 'log_ad_app/test.html')