# BirdWav2Vec

## Requirements
基本的なに必要なライブラリは以下の通り、または、conda環境はリポジトリ内の environment.yml を参照

### 機械学習ライブラリ関連
- pytorch
- sklearn
- numpy
- umap-learn
- joblib

### huggingface関連
- transformers
- datasets
- accelerate
- huggingface_hub

## 準備用 `00prepare_dataset.py`
huggingfaceにデータを上げる用のスクリプト（基本的に利用する際はスキップしてよい）

## 学習 `01run_birddb.sh`
wav2vec2の学習（事前学習）を実行するためのスクリプト
（必要なデータはhuggingfaceよりダウンロードされる）

実行
```
sh 01run_birddb.sh
```
## 埋め込み作成 `02save_embedding.py`
学習済みモデルを用いて、ベクトル埋め込みを計算し、各音データのベクトル埋め込みを作成し、保存する。
ここで、一つの音データに対して指定次元数(256次元)x 時間ステップの可変長の埋め込みが生成される点に注意

実行
```
python 02save_embedding.py
```

## メタデータ `03annotate.py`
ひとつ前のベクトル埋め込みに対応した元の音データのメタデータを抽出し、ベクトルとメタデータの対応付けを行う。

実行
```
python 03annotate.py
```

## ベクトル埋め込みを２次元にプロット `04plot_embedding.py`
ベクトル埋め込みのプロット

指定次元数(256次元)x 時間ステップのベクトル埋め込みになっているので、１点はある音のある時刻の埋め込みに対応している

実行
```
python 04plot_embedding.py
```

## 時間平均＆再度プロット `05mean_vector.py`
上記までで計算したベクトルにつて、時間ステップに関して平均をとったベクトルを計算する。
１点はある音の埋め込みに対応している。

実行
```
python 05mean_vector.py
```

## 同じ種類の音に関して平均＆再度プロット `06mean_vector2.py`
上記までで計算したベクトルについて、同じ音の種類（鳥の種類）に関して平均をとったベクトルを計算する。
１点が一つの音の種類（鳥の種類）の埋め込みに対応している。

実行
```
python 06mean_vector2.py
```
 
