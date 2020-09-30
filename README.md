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

