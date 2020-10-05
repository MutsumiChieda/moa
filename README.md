# Mechanisms of Action

コンペの情報についてはdocs/コンペ概要を参照

## 実行手順
準備：コンペデータを`input/lish-moa/`にダウンロードする。
```shell
python script/create_data.py
python script/train.py
```

train.py  
    -t, --tune: ハイパーパラメータチューニングするか  
    -v, --cv: 交差検証するか  

## 環境構築
```shell
conda env create --name moa python=3.7
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install numpy
conda install pandas 
conda install scikit-learn
conda install -c conda-forge jupyterlab
conda install -c trent-b iterative-stratification
```
