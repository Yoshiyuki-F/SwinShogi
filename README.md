# SwinShogi

SwinShogiは、Swin Transformerを使用した将棋AIシステムです。強化学習と深層学習を組み合わせ、高性能な将棋エンジンを目指しています。

## 主な特徴

- **Swin Transformer**による深層学習モデル
- 交差エントロピー損失による方策学習
- モンテカルロ木探索（MCTS）による思考ルーチン
- JAXによる高速化と最適化
- USI（Universal Shogi Interface）プロトコル対応インターフェース
- 自己対局と外部エンジン対戦による評価システム
- 将棋の完全なルール実装（打ち歩詰め禁止、王手判定、詰み判定、千日手検出など）

## プロジェクト構成

```
SwinShogi/
  - config/       # 設定ファイル
  - data/         # 学習・評価用データ
  - scripts/      # 実行スクリプト
  - src/          # ソースコード
    - interface/  # USIインターフェース
    - model/      # Swin Transformerモデル
    - rl/         # 強化学習アルゴリズム
    - shogi/      # 将棋ルール
    - tests/      # テストコード
    - utils/      # ユーティリティ
```

## モジュール構成

### モデル (src/model/)

- **shogi_model.py**: Swin Transformerベースの将棋モデル定義
  - `SwinShogiModel`: メインのモデルクラス。方策（着手確率）と価値（勝率評価）を出力
  - `create_swin_shogi_model`: モデルとパラメータを初期化する関数
  
- **swin_transformer.py**: Swin Transformer実装
  - `WindowAttention`: ウィンドウ内の注意機構
  - `SwinTransformerBlock`: 基本ブロック（MSAとFFNを含む）
  - `BasicLayer`: 複数のブロックとダウンサンプリングを含むレイヤー
  - `PatchEmbed`: 入力画像をパッチに分割して埋め込む
  - `PatchMerging`: パッチのマージとダウンサンプリング処理

### 強化学習 (src/rl/)

- **trainer.py**: モデルトレーニングクラス
  - `Trainer`: 強化学習トレーニングのメインクラス
  - `TrainState`: パラメータと最適化器の状態を管理

- **mcts.py**: モンテカルロ木探索の実装
  - `MCTS`: 探索木を管理し、最適な手を選択
  - `Node`: 探索木のノード表現

- **self_play.py**: 自己対戦による訓練データ生成

### 将棋ルール (src/shogi/)

- **shogi_game.py**: 将棋ゲームのルール実装
- **shogi_pieces.py**: 駒の定義と動きの実装
- **board_encoder.py**: 盤面状態をモデル入力形式にエンコード
- **board_visualizer.py**: 将棋盤の可視化

### ユーティリティ (src/utils/)

- **model_utils.py**: モデル関連の共通ユーティリティ
  - `predict`: 将棋の状態からモデル推論を実行
  - `inference_jit`: JIT最適化された推論関数
  - `PolicyGradientLoss`: 方策勾配法で使用する損失関数集

- **jax_utils.py**: JAX関連のユーティリティ関数
- **performance.py**: パフォーマンス計測と最適化

### テスト (src/tests/)

- **test_swin_shogi.py**: SwinShogiモデルのテスト
- **test_shogi_rules.py**: 将棋ルール実装のテスト
- **test_performance_evaluation.py**: パフォーマンス評価

## 実装内容

1. 方策損失関数：交差エントロピー損失を使用し、MCTSから得られた行動確率分布とモデルの予測確率分布間の距離を最小化
2. モデルテスト：SwinShogiModelの初期化、推論、パラメータの保存・読み込みテスト
3. USIインターフェース：将棋エンジン通信プロトコル対応、SFEN形式対応、外部エンジン対戦機能
4. 評価システム：自己対局評価と外部エンジン対戦評価
5. パフォーマンス最適化：JAXのJIT最適化とバッチ処理による推論速度向上
6. 将棋ルール：打ち歩詰め禁止、王手判定、詰み判定、千日手検出などの完全実装 