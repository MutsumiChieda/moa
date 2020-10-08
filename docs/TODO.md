# TODO

実装すべきアイデアを記述する。  
**Trelloに移行中**
<https://trello.com/b/YqQV6ObI/moa>

## 戦略
> とにかく大量のモデルをハイパーパラメータ調整して訓練、提出して良いモデルを探す。

## すぐやること（やる気のない日に調査）
- [ ] ハイパーパラメータ入力フローを整備する([gin-config](https://github.com/google/gin-config)を活用する？)



## パイプライン構築の参考
| Competition      | URL                                  | Desc.                         |
| ---------------- | ------------------------------------ | ----------------------------- |
| TReNDS 1st Place | https://github.com/DESimakov/TReNDS/ | README.mdのScriptが参考になる |

## 未分類（何に活かせるか調査が必要、思いつき次第TODOを作成）
- ラベルのないサンプルが40%ある
- 4ラベル以上あるサンプルは1%に満たない  

- g番号が隣の特徴は似た相関傾向を持つグループになっていることがある(corrmapより)  
相関の階層クラスタリングで明確になる？  
- g特徴同士、c特徴同士の相関散布図で、2つのクラスタができている組み合わせがある(e.g. g-0 vs g-8)  

- g特徴には、55個の遺伝子ペアで0.8+の相関がある
- c特徴値に頻出する-10は、実質欠損値か、クリップ値の可能性がある  
そのまま扱う？決定木の場合は？それ以外は？

- c特徴で-10以外の値について相関を見る
- c特徴で、`cp_time==24`のデータでのみDose間に分布の差がある  
- ターゲットクラス同士で相関が強いものがある([表](ターゲット相関.md)参照)
  - モデル選択もこれに基づく可能性が高い
  
何らかの効果の区別に使える？
- cp_type間で、g-525などの特徴値分布にシフトがみられる  
- ターゲット陽性/陰性間でシフトがたくさんありそうなので見つける
  - ターゲット「ドーパミン」の陽性/陰性間で、g-526やg-8特徴値の分布にわずかなシフトがみられる
  - ターゲット「ドーパミン」の陽性/陰性間で、c-42特徴値の分布に、極端な負値には陰性しかない
  - 調べればもっとある
- 名寄せが可能 antagonist = inhibitors = blocker, agonists = activators.

### 追加データセットについて
- 基本データセットより疎で、ラベルのないサンプルが80%ある
- クラスごとのMoA数の分布は基本データより全体的に少な目
- MoA数上位はほとんどingibitorクラス


---

## モデル

![](./experiments%20list.png)

### [MAIN] マルチラベル予測
Comming soon


### 過去コンペのモデル
マルチラベル分類は過去にこの[コンペ](https://www.kaggle.com/c/lish-moa/discussion/180092)で行われた。  

- [ ] NNモデル
  - [ ] ResNet10ベースのNN:  
  AdamW、L1正則化、ARD層*、スナップショット**  
  *Adversarially Robust Distillation  
  **snapshot ensemble  
  - [ ] AutoEncoder(U-Net系) + PCA/ICA*  
  *Independent Component Analysis
  - [x] ResNet
  - [x] Densenet121
  - [ ] Transformer
  - [ ] ResNest14d
  - [ ] GIN (Graph Isomorphism Network)
  - [ ] TabNet
  - [ ] UNet
  
- [ ] 木モデル
  - [ ] RGF
  - [ ] RF
  - [ ] LightGBM
- [ ] 単純な線形モデル
  - [ ] Kernel Ridge
  - [ ] Bayesian Ridge
  - [ ] NuSVM
  - [ ] ElasticNet
- [ ] OMP*  
*OrthogonalMatchingPursuit

## データの取り方
- [ ] 追加データを使う/除くで比較
- [ ] 同じサンプルに対して、agonist, antagonistが同時に反応していないか調べる  
同時に反応していれば、それは間違ったデータ
- [ ] nfkb inhibitor, proteasome inhibitorなど、陽性の多いサンプルを独立して学習させる
- [ ] もしくは、アンダーサンプリング・オーバーサンプリングを試す

## 謎解き
- [ ] [クラス名の読み方](./薬学基礎.md)を参照し、クラスに関する有益な情報を抽出する
  - [ ] g特徴でのクラスタリングで、受容体に作用するか酵素に作用するか分類する
  - [ ] ラベル名からagentを除いた名前による情報を得る
    e.g. antiviral agents(抗ウイルス剤)、antidiabetes agents(抗糖尿病薬)  

## 外部データ   
`Chembl and Pubchem`データセットが有用だがかなり独特らしい  
`clue.io`を調べれば十分？
- [x] アクション（最後の単語）を削除すると、ターゲットをタンパク質名として取得できるので、
  タンパク質名でuniprot.orgをスクレイピングし、特徴を追加する  
  ターゲット名があいまいすぎて今のところ特定不可


## 学習手順
- TReNDS 1st Place  
  - [ ] 疑似ラベルの活用
    ```markdown
    1. 学習と同時にfold内でテストデータに対して予測  
    2. 予測確度.3以上のテストデータは予測を疑似ラベルとして学習データと一緒に再学習
    3. テスト・検証データに対して予測
    ```
  - [ ] 部分的な特徴で学習したOOF予測値の使用
  - [ ] スタッキング(Bayesian Ridge, RGF)やそれらの加重総和

- [ ] `['inhibitor', 'antagonist', 'agonist', 'activator', 'agent', 'blocker', 'OTHERS', None]`でマルチラベル分類してから、その中でマルチラベル分類する  
antagonist = inhibitors = blocker  
agonists = activators  
- [ ] アクティブなラベル数を予測して、それに満たない予測結果に関しては、予測したラベル数だけ最大確率のラベルの確率を引き上げる

## 特徴生成
- [ ] 特徴のEDAの値をそのまま特徴エンコーディングする  
e.g. 用量D1のtrt_cpサンプルの中で、処理時間ごとのサンプル数が異なるので、カウント値を特徴する  

- [ ] doseとtimeがそろっている中からg特徴やc特徴をコントロールの平均で引いたものを追加  

## 前処理
- [ ] StandardScaler
- [ ] cell特徴で欠損値として-10があてられているかを調査
単に多峰性の可能性が高い、特徴ごとに見ていく必要がある  
kdeの描画特性にも注意  
- [ ] cell特徴は分布が歪で、変換が必要
- [ ] c特徴とg特徴別々にPCAしたもの  
元の分散×90%の再現にC特徴は12主成分、G特徴は320主成分かかる  
C特徴はPCAで情報が失われにくいが、G特徴にPCAするのは得策とは言えなそう

## 特徴選択
- [ ] g特徴をダイス係数によるクラスタリングで、有効なg特徴を見つける  
一つの行（分子）が細胞に入るように設計されているか、入らないように設計されているかで、化学的性質の分布が異なる  
RDkitを用いて化学的性質の生成に有用  
### 次元削減
- [ ] インクリメンタルPCA  
チャネルを10個のグループに分割し、その中でflatten

## ヒューリスティック
- [ ] Train, Nonscored, Test 一致するデータを探して予測結果に上書き
- [ ] Train Test 分布が異なる場合、テストの異なる列にバイアスを追加して、訓練に近づける  
  - [ ] `train[col], test[col]+b`間のKS検定統計量を最小化
- [x] `ctl_vehicle`のサンプルには薬剤が使用されていないので、ラベルはすべてゼロとする  ([ref](https://www.kaggle.com/c/lish-moa/discussion/180165))  

## アンサンブル  
- 異なるモデルと特徴群でアンサンブル
  - [ ] モデルを複数作成
  - [ ] 特徴群サンプリング
    - [ ] [NNを分割するアプローチ](https://www.kaggle.com/gogo827jz/split-neural-network-approach-tf-keras)：学習後に各NNをアンサンブルする  
- [ ] Blending  
※予測値の平均がそろっている必要がある  
重みの最適化に[Lagrange Multiplier](kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0)という方法がある
- snapshotアンサンブル

## 試さなくてよいアイデア
- 追加データの活用として、すべてのスコア対象をメタ特徴量として予測する
