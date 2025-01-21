import requests
import pandas as pd
import cv2
from ultralytics import YOLO

# Google Street View APIの設定
API_KEY = 'AIzaSyA9XwC0dlDdgHzmoifLo4OtOb-L4vXkeeY'
BASE_URL = 'https://maps.googleapis.com/maps/api/streetview'

# 座標リストのインポート
def load_coordinates_with_heading(file_path):
    """CSVファイルから座標と方位角リストを読み込む"""
    df = pd.read_csv(file_path)
    return df[['latitude', 'longitude', 'heading']].values.tolist()

# Google Street View画像を取得
def fetch_street_view_image_with_heading(lat, lon, heading, save_path):
    """指定された座標と方位角のStreet View画像を取得"""
    params = {
        'location': f'{lat},{lon}',
        'size': '640x640',
        'heading': heading,  # 方位角を指定
        'key': API_KEY,
        'scale' : 2
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path
    else:
        print(f"Failed to fetch image for {lat}, {lon}, heading {heading}")
        return None

# YOLOv11を使用して植木鉢の数を検出
def count_potted_plants(image_path, model):
    """画像から植木鉢の数を検出"""
    results = model(image_path)
    potted_plant_class = 'potted plant'
    pot_count = 0
    for box in results[0].boxes.data:
        class_id = int(box[5])
        if model.names[class_id] == potted_plant_class:
            pot_count += 1
    return pot_count

# メイン処理
def main():
    # 座標と方位角のリストを読み込み
    coordinates = load_coordinates_with_heading('根津二丁目画像取得用.csv')
    
    # YOLOv11モデルのロード
    model = YOLO('yolo11x.pt')  

    # 座標と植木鉢の数のペアを保存するリスト
    results = []

    for i, (lat, lon, heading) in enumerate(coordinates):
        print(f"Processing {i + 1}/{len(coordinates)}: {lat}, {lon}, heading {heading}")

        # 1. 画像を取得
        image_path = fetch_street_view_image_with_heading(lat, lon, heading, f'temp_{i}.jpg')
        if image_path is None:
            continue

        # 2. 植木鉢の数を検出
        pot_count = count_potted_plants(image_path, model)

        # 3. 結果をリストに追加
        results.append({'latitude': lat, 'longitude': lon, 'heading': heading, 'pot_count': pot_count})

    # 結果をデータフレームに変換して保存
    result_df = pd.DataFrame(results)
    result_df.to_csv('results_with_heading.csv2', index=False)
    print("Processing completed. Results saved to results_with_heading.csv")    

if __name__ == "__main__":
    main()
