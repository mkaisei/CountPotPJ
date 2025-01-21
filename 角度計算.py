import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# 入力シェープファイルのパス
input_shapefile = "根津二丁目プロット.shp"
# 出力シェープファイルのパス
output_shapefile = "根津二丁目角度付.shp"
# 新しいフィールド名
angle_field = "NearestAng"

# シェープファイルを読み込む
gdf = gpd.read_file(input_shapefile)

# 投影座標系に変換（必要に応じて適切なEPSGコードに変更）
# 投影座標系は、距離や角度計算を正確に行うために重要
if not gdf.crs.is_projected:
    gdf = gdf.to_crs(epsg=3857)  # Web Mercator などの適切な投影法に変換

# 角度計算用の関数
def calculate_angle(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = np.degrees(np.arctan2(dx, dy))  # 北を0度、右回りで計算
    return angle if angle >= 0 else angle + 360  # 0-360度に正規化

# 各ポイントに対して最も近い他のポイントを見つける
def find_nearest_angle(row, gdf):
    point = row.geometry
    distances = gdf.distance(point)
    distances.iloc[row.name] = np.inf  # 自分自身の距離を無限大に設定
    nearest_idx = distances.idxmin()  # 最近傍のインデックス
    nearest_point = gdf.loc[nearest_idx].geometry
    return calculate_angle(point, nearest_point)

# 新しいフィールドを追加して角度を計算
gdf[angle_field] = gdf.apply(lambda row: find_nearest_angle(row, gdf), axis=1)

# 結果を保存
gdf.to_file(output_shapefile)

print(f"計算完了: 角度がフィールド '{angle_field}' に追加され、結果が保存されました。")
