"""
将棋AIの強化学習フレームワーク
MCTSを使った自己対局と方策・価値ネットワークの学習
"""

import jax
import jax.numpy as jnp
import os
import sys
import time
import flax
import numpy as np
from pathlib import Path
import logging

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# JAX設定
# GPUが利用できない場合はValueErrorを発生させる
try:
    # GPUが利用可能かチェック
    jax.config.update('jax_enable_x64', True)  # 64ビット精度を有効化
    
    # デバイス情報表示
    devices = jax.devices()
    backend = jax.default_backend()
    print("JAX認識デバイス:", devices)
    print("JAXデフォルトバックエンド:", backend)
    
    # CPUが使用されている場合はエラーを発生させる
    if backend == "cpu" or all("cpu" in str(d).lower() for d in devices):
        raise ValueError("CPUが使用されています。GPUが必要です。")
    
    # GPUの使用状況を確認
    print("\nGPU使用状況の確認:")
    
    # 小さな行列積のテスト
    x = jnp.ones((1000, 1000))
    start_time = time.time()
    for _ in range(10):
        y = jnp.dot(x, x)  # 行列積でGPUに負荷をかける
    jax.block_until_ready(y)
    end_time = time.time()
    print(f"小さな行列積10回の実行時間: {(end_time - start_time)*1000:.2f}ms")
    
    # 大きな行列積のテスト（GPUメモリを多く使用）
    print("\nGPUメモリ使用テスト:")
    try:
        # 5000x5000の行列を作成（約200MB）
        large_x = jnp.ones((5000, 5000), dtype=jnp.float32)
        start_time = time.time()
        large_y = jnp.dot(large_x, large_x)  # 約200MBのGPUメモリを使用
        jax.block_until_ready(large_y)
        end_time = time.time()
        print(f"大きな行列積の実行時間: {(end_time - start_time)*1000:.2f}ms")
        
        # さらに大きな行列積のテスト（約800MB）
        larger_x = jnp.ones((10000, 10000), dtype=jnp.float32)
        start_time = time.time()
        larger_y = jnp.dot(larger_x, larger_x)
        jax.block_until_ready(larger_y)
        end_time = time.time()
        print(f"より大きな行列積の実行時間: {(end_time - start_time)*1000:.2f}ms")
        print("GPUメモリテスト成功！")
    except Exception as e:
        print(f"GPUメモリテスト中にエラーが発生しました: {e}")
    
    print("GPUが正常に使用されています！")
    
except Exception as e:
    print(f"JAX設定エラー: {e}")
    print("GPUが必要です。CUDA対応のJAXlibをインストールしてください。")
    import sys
    sys.exit(1)

from src.shogi.board_encoder import encode_board, create_initial_board, visualize_board
from src.model.swin_shogi import SwinShogiModel
from src.rl.mcts import MCTSConfig
from src.rl.rl_config import RLConfig
from config.default_config import MCTS_CONFIG, MODEL_CONFIG, RL_CONFIG, PATHS

# ロギング設定
logger = logging.getLogger(__name__)

class ActorCritic:
    """
    ActorCriticネットワーク
    
    方策（行動確率）と価値（状態の価値）を予測するニューラルネットワーク
    """
    
    def __init__(self, model, params):
        """
        ActorCriticの初期化
        
        Args:
            model: Flaxモデル（SwinShogiModel）
            params: モデルのパラメータ
        """
        self.model = model
        self.params = params
        
        # JIT最適化された推論関数
        self.predict_jit = jax.jit(self._predict)
    
    def predict(self, state_features):
        """
        状態から方策と価値を予測する
        
        Args:
            state_features: 状態の特徴量（盤面のエンコーディングなど）
            
        Returns:
            (行動確率の辞書, 状態の価値)
        """
        # 推論用の形状に変換（バッチ次元を追加）
        if len(state_features.shape) == 3:
            state_features = jnp.expand_dims(state_features, axis=0)
            
        # 推論実行
        policy_logits, value = self.predict_jit(self.params, state_features)
        
        # バッチ次元を削除
        policy_logits = policy_logits[0]
        value = value[0][0]
        
        # 方策ロジットを確率に変換
        policy_probs = jax.nn.softmax(policy_logits)
        
        # 確率を辞書形式に変換
        # ここでは簡略化のため、インデックスから行動へのマッピングは実装していない　TODO
        # 実際には駒の移動やその他のゲーム固有の行動にマッピングする必要がある
        action_probs = {i: float(prob) for i, prob in enumerate(policy_probs) if prob > 0.001}
        
        return action_probs, float(value)
    
    def _predict(self, params, state_features):
        """
        JIT最適化用の内部予測関数
        """
        return self.model.apply(params, state_features)
    
    def update(self, new_params):
        """
        パラメータを更新する
        
        Args:
            new_params: 新しいパラメータ
        """
        self.params = new_params

class PolicyGradientLoss:
    """
    方策勾配損失関数
    
    方策勾配法（REINFORCE、A2Cなど）で使用する損失関数を実装します。
    """
    
    @staticmethod
    def policy_loss(logits, action_probs, advantages):
        """
        方策損失を計算する
        
        Args:
            logits: モデルの出力ロジット
            action_probs: ターゲット行動確率
            advantages: アドバンテージ値
            
        Returns:
            方策損失値
        """
        # 交差エントロピー損失
        policy_loss = -jnp.sum(action_probs * jax.nn.log_softmax(logits) * advantages) / logits.shape[0]
        return policy_loss
    
    @staticmethod
    def value_loss(values, target_values):
        """
        価値損失を計算する
        
        Args:
            values: モデルが予測した価値
            target_values: ターゲット価値
            
        Returns:
            価値損失値
        """
        # 二乗誤差
        value_loss = jnp.mean(jnp.square(values - target_values))
        return value_loss
    
    @staticmethod
    def entropy_loss(logits):
        """
        エントロピー損失を計算する
        
        Args:
            logits: モデルの出力ロジット
            
        Returns:
            エントロピー損失値
        """
        # エントロピーの計算（高いほど探索的）
        probabilities = jax.nn.softmax(logits)
        log_probabilities = jax.nn.log_softmax(logits)
        entropy = -jnp.sum(probabilities * log_probabilities, axis=1)
        return -jnp.mean(entropy)  # 最大化したいので負の値に
    
    @staticmethod
    def total_loss(policy_loss, value_loss, entropy_loss, entropy_coeff=0.01):
        """
        全体の損失を計算する
        
        Args:
            policy_loss: 方策損失
            value_loss: 価値損失
            entropy_loss: エントロピー損失
            entropy_coeff: エントロピー項の係数
            
        Returns:
            全体の損失値
        """
        return policy_loss + value_loss + entropy_coeff * entropy_loss

class TrainState:
    """
    トレーニング状態
    
    モデルのパラメータと最適化器の状態を管理します。
    """
    
    def __init__(self, params, optimizer_state=None):
        """
        トレーニング状態の初期化
        
        Args:
            params: モデルのパラメータ
            optimizer_state: 最適化器の状態
        """
        self.params = params
        self.optimizer_state = optimizer_state
        self.step = 0
        
    def update(self, params, optimizer_state):
        """
        状態を更新する
        
        Args:
            params: 新しいモデルパラメータ
            optimizer_state: 新しい最適化器状態
            
        Returns:
            更新された状態
        """
        self.params = params
        self.optimizer_state = optimizer_state
        self.step += 1
        return self

class Trainer:
    """
    モデルのトレーニングクラス
    
    SwinShogiモデルを強化学習でトレーニングするためのクラスです。
    """
    
    def __init__(
        self, 
        model, 
        initial_params,
        learning_rate=0.001,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        entropy_coeff=0.01,
        save_dir="data/models"
    ):
        """
        トレーナーの初期化
        
        Args:
            model: トレーニングするモデル
            initial_params: 初期パラメータ
            learning_rate: 学習率
            weight_decay: 重み減衰
            max_grad_norm: 勾配クリッピングの最大ノルム
            entropy_coeff: エントロピー項の係数
            save_dir: モデルを保存するディレクトリ
        """
        self.model = model
        self.train_state = TrainState(initial_params)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.entropy_coeff = entropy_coeff
        self.save_dir = Path(save_dir)
        
        # 保存ディレクトリの作成
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 最適化器の設定
        self.optimizer = flax.optim.Adam(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # 最適化器の初期化
        self.optimizer_state = self.optimizer.init(initial_params)
        
        # トレーニングステップのJIT最適化
        self.train_step_jit = jax.jit(self._train_step)
        
        # トレーニング統計
        self.stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "grad_norm": []
        }
    
    def _train_step(self, params, optimizer_state, states, actions, advantages, target_values):
        """
        1トレーニングステップを実行する
        
        Args:
            params: モデルパラメータ
            optimizer_state: 最適化器の状態
            states: 状態のバッチ
            actions: 行動のバッチ
            advantages: アドバンテージのバッチ
            target_values: ターゲット価値のバッチ
            
        Returns:
            (新しいパラメータ, 新しい最適化器状態, 損失値の辞書)
        """
        def loss_fn(params):
            """勾配を計算するための損失関数"""
            # モデルの順伝播
            policy_logits, values = self.model.apply(params, states)
            
            # アドバンテージと行動をone-hotエンコーディングに変換
            action_onehot = jax.nn.one_hot(actions, policy_logits.shape[1])
            
            # 損失の計算
            policy_loss = PolicyGradientLoss.policy_loss(policy_logits, action_onehot, advantages)
            value_loss = PolicyGradientLoss.value_loss(values, target_values)
            entropy_loss = PolicyGradientLoss.entropy_loss(policy_logits)
            total_loss = PolicyGradientLoss.total_loss(
                policy_loss, value_loss, entropy_loss, self.entropy_coeff)
                
            return total_loss, (policy_loss, value_loss, entropy_loss)
        
        # 勾配の計算
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (policy_loss, value_loss, entropy_loss)), grads = grad_fn(params)
        
        # 勾配のノルムを計算
        flat_grads = jax.tree_leaves(grads)
        global_norm = jnp.sqrt(jnp.sum(jnp.array([jnp.sum(jnp.square(g)) for g in flat_grads])))
        
        # 勾配クリッピング
        grads = jax.tree_map(
            lambda g: jnp.clip(g, -self.max_grad_norm, self.max_grad_norm),
            grads
        )
        
        # パラメータの更新
        updates, new_optimizer_state = self.optimizer.update(grads, optimizer_state, params)
        new_params = flax.optim.apply_updates(params, updates)
        
        # 損失値を辞書に格納
        losses = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": total_loss,
            "grad_norm": global_norm
        }
        
        return new_params, new_optimizer_state, losses
    
    def train_batch(self, states, actions, advantages, target_values):
        """
        バッチデータでトレーニングを実行する
        
        Args:
            states: 状態のバッチ（形状: [batch_size, height, width, channels]）
            actions: 行動のバッチ（形状: [batch_size]）
            advantages: アドバンテージのバッチ（形状: [batch_size]）
            target_values: ターゲット価値のバッチ（形状: [batch_size, 1]）
            
        Returns:
            損失値の辞書
        """
        # トレーニングステップの実行
        new_params, new_optimizer_state, losses = self.train_step_jit(
            self.train_state.params,
            self.optimizer_state,
            states,
            actions,
            advantages,
            target_values
        )
        
        # トレーニング状態の更新
        self.train_state = self.train_state.update(new_params, new_optimizer_state)
        self.optimizer_state = new_optimizer_state
        
        # 統計情報の更新
        for key, value in losses.items():
            self.stats[key].append(float(value))
        
        return losses
    
    def save_model(self, filename=None):
        """
        モデルを保存する
        
        Args:
            filename: 保存するファイル名（Noneの場合は自動生成）
            
        Returns:
            保存先のパス
        """
        if filename is None:
            filename = f"model_step_{self.train_state.step}.pkl"
            
        save_path = self.save_dir / filename
        self.model.save_params(str(save_path), self.train_state.params)
        
        logger.info(f"モデルを保存しました: {save_path}")
        return save_path
    
    def load_model(self, filepath):
        """
        モデルを読み込む
        
        Args:
            filepath: 読み込むファイルパス
            
        Returns:
            読み込みが成功したかどうか
        """
        try:
            params = self.model.load_params(filepath)
            self.train_state = TrainState(params, self.optimizer_state)
            logger.info(f"モデルを読み込みました: {filepath}")
            return True
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False

class SelfPlay:
    """
    自己対戦クラス
    
    モデルを使用して自己対戦を行い、トレーニングデータを生成します。
    """
    
    def __init__(self, game_env, actor_critic, mcts_simulations=400, temperature=1.0):
        """
        自己対戦の初期化
        
        Args:
            game_env: ゲーム環境
            actor_critic: ActorCriticネットワーク
            mcts_simulations: MCTSのシミュレーション回数
            temperature: 行動選択の温度パラメータ
        """
        self.game_env = game_env
        self.actor_critic = actor_critic
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
    
    def play_game(self):
        """
        1ゲームの自己対戦を実行する
        
        Returns:
            (状態のリスト, 行動のリスト, 報酬のリスト, 最終結果)
        """
        from src.rl.mcts import MCTS
        
        # ゲームの初期化
        game = self.game_env.clone()
        game.reset()
        
        # MCTSの初期化
        mcts = MCTS(game, self.actor_critic)
        
        # 対局データの保存用
        states = []
        actions = []
        rewards = []
        
        # 最大手数（将棋は通常300手程度）
        max_moves = 500
        
        for move_num in range(max_moves):
            # 現在の状態を保存
            state_features = game.get_features()
            states.append(state_features)
            
            # MCTSで行動を選択
            for _ in range(self.mcts_simulations):
                mcts.search()
                
            # 温度パラメータを調整（序盤は探索的、終盤は最適）
            if move_num < 30:
                temp = self.temperature
            else:
                temp = 0.1
                
            # 行動確率の取得
            action, action_probs = mcts.get_action_probabilities(temperature=temp)
            
            # 行動を実行
            game.move(action)
            actions.append(action)
            
            # 報酬（勝敗）を確認
            if game.is_terminal():
                result = game.get_result(game.current_player)
                rewards.extend([result] * len(states))
                break
            
            # 報酬は終了時以外は0
            rewards.append(0.0)
            
            # 探索木を更新
            mcts.update_with_move(action)
            
        return states, actions, rewards, game.get_result(0)  # 先手視点の結果
    
    def generate_training_data(self, num_games=10):
        """
        複数ゲームの自己対戦を実行し、トレーニングデータを生成する
        
        Args:
            num_games: 対局数
            
        Returns:
            (状態のバッチ, 行動のバッチ, 報酬のバッチ)
        """
        all_states = []
        all_actions = []
        all_rewards = []
        results = []
        
        for i in range(num_games):
            start_time = time.time()
            states, actions, rewards, result = self.play_game()
            
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            results.append(result)
            
            end_time = time.time()
            logger.info(f"ゲーム {i+1}/{num_games} 完了 - 所要時間: {end_time - start_time:.1f}秒, 結果: {result}")
        
        # 統計情報
        win_rate = results.count(1.0) / len(results)
        draw_rate = results.count(0.0) / len(results)
        loss_rate = results.count(-1.0) / len(results)
        
        logger.info(f"自己対戦結果 - 勝率: {win_rate:.2f}, 引分率: {draw_rate:.2f}, 敗率: {loss_rate:.2f}")
        
        return np.array(all_states), np.array(all_actions), np.array(all_rewards)

def main():
    # シード固定
    key = jax.random.PRNGKey(42)
    
    print("reinforcement_learning.py 内の main 関数は ShogiRL を使用するため無効化されています。")
    print("board_encoder.py が実装され、visualize_board 関数も追加されました。")
    
    # テスト：初期盤面を生成して表示
    print("\n初期盤面のテスト:")
    initial_board = create_initial_board()
    visualize_board(initial_board)
    
    """
    # モデルの初期化 - MODEL_CONFIGから設定を読み込む
    model = SwinShogiModel(
        embed_dim=MODEL_CONFIG['embed_dim'],
        depths=MODEL_CONFIG['depths'],
        num_heads=MODEL_CONFIG['num_heads'],
        window_size=MODEL_CONFIG['window_size'],
        mlp_ratio=MODEL_CONFIG['mlp_ratio'],
        drop_rate=MODEL_CONFIG['dropout'],
        n_policy_outputs=MODEL_CONFIG['policy_dim']
    )
    # 初期盤面を作成（モデル初期化用）
    features = encode_board(create_initial_board())
    print(f"エンコードされた特徴量の形状: {features.shape}")
    # (channels, height, width) -> (height, width, channels) に転置
    features = jnp.transpose(features, (1, 2, 0))
    print(f"転置後の特徴量の形状: {features.shape}")
    features = jnp.expand_dims(features, axis=0)
    print(f"バッチ次元追加後の特徴量の形状: {features.shape}")
    params = model.init(key, features)
    """

    # 強化学習の設定 - MCTS_CONFIGから設定を読み込む
    from src.rl.mcts import MCTSConfig
    
    # デフォルト設定を使用
    mcts_config = MCTSConfig(
        simulation_times=MCTS_CONFIG['simulation_times'],
        expansion_threshold=MCTS_CONFIG['expansion_threshold'],
        gamma=MCTS_CONFIG['gamma'],
        uct_c=MCTS_CONFIG['uct_c'],
        dirichlet_alpha=MCTS_CONFIG['dirichlet_alpha'],
        dirichlet_weight=MCTS_CONFIG['dirichlet_weight']
    )
    
    print(f"\nMCTS設定:")
    for key, value in MCTS_CONFIG.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 