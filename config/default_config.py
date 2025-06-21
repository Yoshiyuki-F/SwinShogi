"""
SwinShogiのデフォルト設定
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Callable
import flax.linen as nn

# モデル設定
MODEL_CONFIG = {
    'img_size': (9, 9),             # 将棋盤のサイズ
    'patch_size': (1, 1),           # パッチサイズ
    'in_chans': 2,                  # 入力チャネル数（駒の種類とプレイヤー）
    'embed_dim': 96,                # 埋め込み次元
    'depths': (3, 3),               # 各段階の層の深さ
    'num_heads': (3, 6),            # 各段階のヘッド数
    'window_size': 3,               # 注意機構のウィンドウサイズ
    'mlp_ratio': 4.0,               # MLPの拡大率
    'qkv_bias': True,               # QKV投影にバイアスを使うかどうか
    'drop_rate': 0.0,               # ドロップアウト率
    'attn_drop_rate': 0.0,          # 注意機構のドロップアウト率
    'drop_path_rate': 0.1,          # パスドロップ率
    'norm_layer': nn.LayerNorm,     # 正規化層
    'patch_norm': True,             # パッチ正規化を行うかどうか
    'n_policy_outputs': 9*9*27,     # 政策出力（移動元×移動先×駒の種類）
    'use_checkpoint': False,        # チェックポイントを使用するかどうか
    'shift_size': 0,                # シフトサイズ（SW-MSA用）
    'downsample': None,             # ダウンサンプリング関数
    'default_img_size': (224, 224), # デフォルト画像サイズ（標準的なSwin Transformer用）
    'default_patch_size': (4, 4),   # デフォルトパッチサイズ（標準的なSwin Transformer用）
    'default_in_chans': 3,          # デフォルト入力チャネル数（標準的なSwin Transformer用）
    'drop': 0.0,                    # 一般的なドロップアウト率
    'drop_path': 0.0,               # パスドロップ率（個別ブロック用）
    'input_resolution': (9, 9),     # 入力解像度
    'patch_merge_factor': 3,        # パッチ結合の分割係数（通常のSwinは2、このSwinは3）
    'feature_dim': 15               # 特徴ベクトル（手番と持ち駒）の次元
}

# 強化学習設定
RL_CONFIG = {
    'num_iterations': 10,
    'num_episodes': 10,
    'max_moves': 300,
    'batch_size': 32,
    'learning_rate': 0.001,
    'value_coef': 1.0,
    'entropy_coef': 0.01
}

# MCTS設定
MCTS_CONFIG = {
    'simulation_times': 100,
    'expansion_threshold': 1,
    'gamma': 1.0,
    'uct_c': 1.5,
    'dirichlet_alpha': 0.3,
    'dirichlet_weight': 0.25,
    'value_weight': 0.5,
    'action_num': 2187,
    'n_simulations': 400,
    'exploration_fraction': 0.25,
    'pb_c_init': 1.25,
    'pb_c_base': 19652,
    'temperature': 1.0,
    'temperature_drop_step': 30,
    'discount': 1.0,
    'value_threshold': -float('inf'),
    'visit_threshold': 1,
    'virtual_loss': 3
}

# USI設定
USI_CONFIG = {
    'engine_name': 'SwinShogi',
    'author': 'yoshi',
    'thinking_time': 5000,  # ミリ秒
    'byoyomi': 1000,  # ミリ秒
}

# ファイルパス
PATHS = {
    'model_params': 'data/trained_params.msgpack',
    'log_dir': 'data/logs',
    'replay_buffer': 'data/replay_buffer.pkl',
    'model_dir': 'data/models',
    'data_dir': 'data/games'
}

# モデルアーキテクチャ設定
@dataclass
class ModelConfig:
    """SwinShogiモデルの設定パラメータ"""
    img_size: Tuple[int, int] = MODEL_CONFIG['img_size']
    patch_size: Tuple[int, int] = MODEL_CONFIG['patch_size']
    in_chans: int = MODEL_CONFIG['in_chans']
    embed_dim: int = MODEL_CONFIG['embed_dim']
    depths: Tuple[int, ...] = MODEL_CONFIG['depths']
    num_heads: Tuple[int, ...] = MODEL_CONFIG['num_heads']
    window_size: int = MODEL_CONFIG['window_size']
    mlp_ratio: float = MODEL_CONFIG['mlp_ratio']
    qkv_bias: bool = MODEL_CONFIG['qkv_bias']
    drop_rate: float = MODEL_CONFIG['drop_rate']
    attn_drop_rate: float = MODEL_CONFIG['attn_drop_rate']
    drop_path_rate: float = MODEL_CONFIG['drop_path_rate']
    norm_layer: Callable = MODEL_CONFIG['norm_layer']
    patch_norm: bool = MODEL_CONFIG['patch_norm']
    n_policy_outputs: int = MODEL_CONFIG['n_policy_outputs']
    use_checkpoint: bool = MODEL_CONFIG['use_checkpoint']
    shift_size: int = MODEL_CONFIG['shift_size']
    downsample: Optional[Callable] = MODEL_CONFIG['downsample']
    default_img_size: Tuple[int, int] = MODEL_CONFIG['default_img_size']
    default_patch_size: Tuple[int, int] = MODEL_CONFIG['default_patch_size']
    default_in_chans: int = MODEL_CONFIG['default_in_chans']
    drop: float = MODEL_CONFIG['drop']
    drop_path: float = MODEL_CONFIG['drop_path']
    input_resolution: Tuple[int, int] = MODEL_CONFIG['input_resolution']
    patch_merge_factor: int = MODEL_CONFIG['patch_merge_factor']
    feature_dim: int = MODEL_CONFIG['feature_dim']

# データクラス定義（rl_config.pyからマージ）

@dataclass
class MCTSConfig:
    """MCTSの設定パラメータ"""
    
    # 探索設定
    c_puct: float = MCTS_CONFIG['uct_c']       # 探索と活用のバランスを調整する係数
    n_simulations: int = MCTS_CONFIG['n_simulations']  # 1回の意思決定で実行するシミュレーション回数
    dirichlet_alpha: float = MCTS_CONFIG['dirichlet_alpha']  # ルートノイズのディリクレ分布のアルファパラメータ
    exploration_fraction: float = MCTS_CONFIG['exploration_fraction']  # ルートノイズの割合
    
    # バックアップ設定
    discount: float = MCTS_CONFIG['discount']   # 割引率（将棋では1.0が一般的）
    
    # ノード展開設定
    value_threshold: float = MCTS_CONFIG['value_threshold']  # 価値がこの閾値以上のノードのみ展開する
    visit_threshold: int = MCTS_CONFIG['visit_threshold']    # この訪問回数以上のノードのみ展開する
    
    # その他の設定
    virtual_loss: int = MCTS_CONFIG['virtual_loss']         # 並列探索用の仮想損失
    temperature: float = MCTS_CONFIG['temperature']         # 行動選択の温度パラメータ
    temperature_drop_step: int = MCTS_CONFIG['temperature_drop_step']  # 温度を下げる手数
    action_num: int = MCTS_CONFIG['action_num']             # 行動空間のサイズ
    simulation_times: int = MCTS_CONFIG['simulation_times'] # シミュレーション回数
    expansion_threshold: int = MCTS_CONFIG['expansion_threshold'] # 展開閾値
    gamma: float = MCTS_CONFIG['gamma']                     # 割引率
    dirichlet_weight: float = MCTS_CONFIG['dirichlet_weight'] # ディリクレノイズの重み
    value_weight: float = MCTS_CONFIG['value_weight']       # 価値と方策の重み付け
    pb_c_init: float = MCTS_CONFIG['pb_c_init']             # PUCT計算の初期係数
    pb_c_base: int = MCTS_CONFIG['pb_c_base']               # PUCT計算のベース値

@dataclass
class TrainingConfig:
    """モデルトレーニングの設定パラメータ"""
    
    # 一般的な設定
    batch_size: int = RL_CONFIG['batch_size']        # バッチサイズ
    num_epochs: int = 10                             # エポック数
    learning_rate: float = RL_CONFIG['learning_rate'] # 学習率
    weight_decay: float = 1e-4                       # L2正則化の係数
    
    # 損失関数の設定
    value_loss_coeff: float = RL_CONFIG['value_coef'] # 価値損失の係数
    policy_loss_coeff: float = 1.0                   # 方策損失の係数
    entropy_coeff: float = RL_CONFIG['entropy_coef'] # エントロピー損失の係数
    
    # 最適化設定
    max_grad_norm: float = 0.5                       # 勾配クリッピングの最大ノルム
    
    # スケジューラ設定
    lr_scheduler: bool = True                        # 学習率スケジューラーを使うかどうか
    lr_scheduler_gamma: float = 0.95                 # 学習率減衰率
    lr_scheduler_step_size: int = 1000               # 何ステップごとに学習率を減衰させるか

@dataclass
class SelfPlayConfig:
    """自己対戦の設定パラメータ"""
    
    # 対戦設定
    num_games: int = RL_CONFIG['num_episodes']       # 生成するゲーム数
    max_moves: int = RL_CONFIG['max_moves']          # 1ゲームあたりの最大手数
    
    # データ生成設定
    num_workers: int = 4                             # 並列ワーカー数
    buffer_size: int = 100000                        # リプレイバッファのサイズ
    save_interval: int = 100                         # データを保存する間隔（ゲーム数）
    
    # 評価設定
    eval_interval: int = 100                         # モデル評価の間隔（トレーニングステップ数）
    eval_games: int = 40                             # 評価時の対局数

@dataclass
class RLConfig:
    """強化学習の全体設定"""
    
    # コンポーネント設定
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # ロギング設定
    log_interval: int = 10                           # ログを出力する間隔（トレーニングステップ数）
    save_model_interval: int = 100                   # モデルを保存する間隔（トレーニングステップ数）
    checkpoint_interval: int = 1000                  # チェックポイントを保存する間隔（トレーニングステップ数）
    
    # パス設定
    model_dir: str = PATHS['model_dir']              # モデルを保存するディレクトリ
    data_dir: str = PATHS['data_dir']                # ゲームデータを保存するディレクトリ
    log_dir: str = PATHS['log_dir']                  # ログを保存するディレクトリ
    
    # その他の設定
    resume_training: bool = True                     # トレーニングを再開するかどうか
    resume_model_path: Optional[str] = None          # 再開するモデルのパス
    random_seed: int = 42                            # 乱数シード
    num_iterations: int = RL_CONFIG['num_iterations'] # 強化学習の反復回数

def get_default_config() -> RLConfig:
    """デフォルト設定を取得する"""
    return RLConfig()

def get_model_config() -> ModelConfig:
    """モデル設定を取得する"""
    return ModelConfig()

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
            if isinstance(value, dict) and isinstance(getattr(config, key), (MCTSConfig, TrainingConfig, SelfPlayConfig, ModelConfig)):
                # ネストされた設定の更新
                nested_config = getattr(config, key)
                for nested_key, nested_value in value.items():
                    if hasattr(nested_config, nested_key):
                        setattr(nested_config, nested_key, nested_value)
            else:
                # トップレベルの設定の更新
                setattr(config, key, value)
    
    return config 