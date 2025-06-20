"""
SwinShogiのデフォルト設定
"""

# モデル設定
MODEL_CONFIG = {
    'embed_dim': 96,
    'depths': (2, 2),
    'num_heads': (8, 16),
    'window_size': 3,
    'mlp_ratio': 4.0,
    'dropout': 0.0,
    'policy_dim': 2187
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
    'dirichlet_weight': 0.25
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
    'replay_buffer': 'data/replay_buffer.pkl'
} 