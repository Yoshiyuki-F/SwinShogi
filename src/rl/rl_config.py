"""
強化学習の設定ファイル

このモジュールでは、MCTSやトレーニングのハイパーパラメータなどの設定を定義します。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class MCTSConfig:
    """MCTSの設定パラメータ"""
    
    # 探索設定
    c_puct: float = 1.5              # 探索と活用のバランスを調整する係数
    n_simulations: int = 800         # 1回の意思決定で実行するシミュレーション回数
    dirichlet_alpha: float = 0.25    # ルートノイズのディリクレ分布のアルファパラメータ
    exploration_fraction: float = 0.25  # ルートノイズの割合
    
    # バックアップ設定
    discount: float = 1.0            # 割引率（将棋では1.0が一般的）
    
    # ノード展開設定
    value_threshold: float = -float('inf')  # 価値がこの閾値以上のノードのみ展開する
    visit_threshold: int = 1          # この訪問回数以上のノードのみ展開する
    
    # その他の設定
    virtual_loss: int = 3            # 並列探索用の仮想損失
    temperature: float = 1.0         # 行動選択の温度パラメータ
    temperature_drop_step: int = 30  # 温度を下げる手数

@dataclass
class TrainingConfig:
    """モデルトレーニングの設定パラメータ"""
    
    # 一般的な設定
    batch_size: int = 512            # バッチサイズ
    num_epochs: int = 10             # エポック数
    learning_rate: float = 2e-4      # 学習率
    weight_decay: float = 1e-4       # L2正則化の係数
    
    # 損失関数の設定
    value_loss_coeff: float = 1.0    # 価値損失の係数
    policy_loss_coeff: float = 1.0   # 方策損失の係数
    entropy_coeff: float = 0.01      # エントロピー損失の係数
    
    # 最適化設定
    max_grad_norm: float = 0.5       # 勾配クリッピングの最大ノルム
    
    # スケジューラ設定
    lr_scheduler: bool = True        # 学習率スケジューラーを使うかどうか
    lr_scheduler_gamma: float = 0.95  # 学習率減衰率
    lr_scheduler_step_size: int = 1000  # 何ステップごとに学習率を減衰させるか

@dataclass
class SelfPlayConfig:
    """自己対戦の設定パラメータ"""
    
    # 対戦設定
    num_games: int = 1000            # 生成するゲーム数
    max_moves: int = 512             # 1ゲームあたりの最大手数
    
    # データ生成設定
    num_workers: int = 4             # 並列ワーカー数
    buffer_size: int = 100000        # リプレイバッファのサイズ
    save_interval: int = 100         # データを保存する間隔（ゲーム数）
    
    # 評価設定
    eval_interval: int = 100         # モデル評価の間隔（トレーニングステップ数）
    eval_games: int = 40             # 評価時の対局数

@dataclass
class RLConfig:
    """強化学習の全体設定"""
    
    # コンポーネント設定
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    
    # ロギング設定
    log_interval: int = 10           # ログを出力する間隔（トレーニングステップ数）
    save_model_interval: int = 100   # モデルを保存する間隔（トレーニングステップ数）
    checkpoint_interval: int = 1000  # チェックポイントを保存する間隔（トレーニングステップ数）
    
    # パス設定
    model_dir: str = "data/models"   # モデルを保存するディレクトリ
    data_dir: str = "data/games"     # ゲームデータを保存するディレクトリ
    log_dir: str = "logs"            # ログを保存するディレクトリ
    
    # その他の設定
    resume_training: bool = True     # トレーニングを再開するかどうか
    resume_model_path: Optional[str] = None  # 再開するモデルのパス
    random_seed: int = 42            # 乱数シード

def get_default_config() -> RLConfig:
    """デフォルト設定を取得する"""
    return RLConfig()

def update_config(config: RLConfig, updates: Dict[str, Any]) -> RLConfig:
    """
    設定を更新する
    
    Args:
        config: 元の設定
        updates: 更新する項目の辞書
        
    Returns:
        更新された設定
    """
    for key, value in updates.items():
        if hasattr(config, key):
            if isinstance(value, dict) and isinstance(getattr(config, key), (MCTSConfig, TrainingConfig, SelfPlayConfig)):
                # ネストされた設定の更新
                nested_config = getattr(config, key)
                for nested_key, nested_value in value.items():
                    if hasattr(nested_config, nested_key):
                        setattr(nested_config, nested_key, nested_value)
            else:
                # トップレベルの設定の更新
                setattr(config, key, value)
    
    return config 