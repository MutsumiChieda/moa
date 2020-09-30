# Abhishek Thakur + Andrey Lukyanenkoのペアプログラミング
EDA、モデリング、

マルチラベルクラス分類

説明を読む  
gは一般情報 cは細胞情報 

erbb2, diureticなど陽性が少ないので0予測してよさそう
nfkb_ingibitor, protesome_inhibitorなど陽性の多い特徴がかなり重要になる

データ分布が似ているので、shakeupはあまり起こらなそう

g特徴値の分布は正規分布に近い  
c特徴値に実質の欠損値(e.g. -10)がありそう  

pytorch-lightningを使う

