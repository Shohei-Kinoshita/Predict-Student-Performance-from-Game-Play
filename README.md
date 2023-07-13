## Predict-Student-Performance-from-Game-Play(Kaggle)
### 77th Solution

### コンペ概要
 - ゲームベースの英語学習教材を、プレイした生徒のログデータから成績(ゲーム内の質問に回答できるか)を予測するコンペ
 - ゲームは3つのセクションから構成されていて、予測時はセクション1のデータ→セクション2のデータ→セクション3のデータの順に送られ、各セクションごとに予測をしなければいけない
   - セクション1：Q1~Q3
   - セクション2：Q4~Q13
   - セクション3：Q14~Q18
 - 合計18個の質問に対して正解したかどうか(0 or 1)を予測
 - コンペ詳細(https://www.kaggle.com/competitions/predict-student-performance-from-game-play)

#### モデル
 - XGBoostとCatboostのアンサンブル(7:3でblend)

#### バリデーション
 - KFold(5fold)でCVのスコアを確認し、その後全データを用いて学習(n_estimatorは各foldの中央値を使用)

#### 戦略
今回のコンペは、3つのセクションから構成された合計18個の質問に`正解したかどうか(0 or 1)`を予測するコンペのため、各質問を予測する合計18のモデルを作成<br>
また、サブミット時はTime Series APIというKaggle独自のAPIが使われており、各セクションのデータが送られそのセクションに対応した質問の予測をしないと次のデータが送られない<br>
※つまり、未来のデータを使用して予測ができない。<br>
逆に言うと、セクション2の時点ではセクション1のデータも使うことができるため、セクション2以降はそれまでのログデータをすべて縦積みして特徴量を作成した

#### 特徴量
このコンペはユーザーに対して予測を行うため、ユーザーごとに集約関数(sum, max, diff等々)を適用し特徴量を作成<br>
また、今回は`ゲームをプレイするのにどれだけ時間がかかったか`が重要だったため、ある時点からある時点までの経過時間の特徴量が一番効いた<br>
かなりたくさんの特徴量を作成したが、基本的な考えは以下の通り
 - セクション全体に対するプレイ時間
 - セクション内の各イベントをクリアするまでのプレイ時間
 - 以前に予測している確率を特徴量として使う(例えば、10問目の予測時には1~9問目の予測値を使う)
 
#### 欠損値補完
 - Nullのまま使用
   
#### ハイパーパラメーターチューニング
 - optunaにてチューニング<br>
 ※hold-outでのチューニングだと過学習となったため、全データに対するoofを最適にするようにチューニングした(1trialに1h近くかかった...)
 
#### 試したがうまくいかなったこと
 - LSTM
   - 時系列データとして学習するモデルを試したが、GBDT以上の精度が出なかった
 - 学習データの追加
   - 公開されている生データを学習に追加したが、精度が変わらなかった<br>
     ※生データを使って精度を上げている人がいたため、ここはもっと工夫するべきだった
 - 閾値の最適化
   - 各予測値に対して最適な閾値を求めたが、精度が下がった<br>
     ※今回のコンペはmacroF1だったため、単純な各予測値を最適化するだけではダメだった
 
#### 試せばよかったこと、その他反省
 - 学習の高速化
   - 1回の実験で2h近くかかっていたため高速化の工夫をするべきだった
     ※特徴量選択を行うなど
  
#### その他
 - ログデータを見るだけ(EDA等)では、仮説を立てることも難しかったため実際にゲームをプレイしながら仮説を立てたことが良い特徴量作成につながった<br>
