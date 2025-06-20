"""
将棋AIの強化学習フレームワーク
MCTSを使った自己対局と方策・価値ネットワークの学習
"""

import os
import sys
import logging

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 分割したモジュールをインポート
from src.utils.jax_utils import setup_jax, create_rng_keys
from src.model.actor_critic import ActorCritic
from src.rl.trainer import Trainer, TrainState, PolicyGradientLoss
from src.rl.self_play import SelfPlay

from src.shogi.board_encoder import create_initial_board, visualize_board
from config.default_config import MCTS_CONFIG

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reinforcement_learning.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    # JAX設定
    if not setup_jax(require_gpu=True):
        logger.error("JAXの設定に失敗しました。終了します。")
        return
    
    # テスト：初期盤面を生成して表示
    print("\n初期盤面のテスト:")
    initial_board = create_initial_board()
    visualize_board(initial_board)
    
    print(f"\nMCTS設定:")
    for key, value in MCTS_CONFIG.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 