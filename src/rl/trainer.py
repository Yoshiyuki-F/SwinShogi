"""
強化学習のトレーニング関連クラス
"""

import jax
import jax.numpy as jnp
import os
import flax
import logging
from pathlib import Path
from src.utils.model_utils import PolicyGradientLoss, process_gradients

# ロギング設定
logger = logging.getLogger(__name__)

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
            
            # utils内の総合損失計算関数を使用
            return PolicyGradientLoss.compute_losses_from_model_outputs(
                policy_logits, values, action_onehot, advantages, target_values, self.entropy_coeff)
        
        # 勾配の計算
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (policy_loss, value_loss, entropy_loss)), grads = grad_fn(params)
        
        # 勾配の処理（ノルム計算とクリッピング）
        clipped_grads, global_norm = process_gradients(grads, self.max_grad_norm)
        
        # パラメータの更新
        updates, new_optimizer_state = self.optimizer.update(clipped_grads, optimizer_state, params)
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

# 訓練と推論関連の基本機能はsrc/utils/model_utils.pyに集約されています

if __name__ == "__main__":
    # テスト実行コード - 実際のトレーニング開始スクリプトは別途作成します
    import argparse
    
    parser = argparse.ArgumentParser(description="将棋強化学習トレーナー")
    parser.add_argument("--resume", type=str, help="トレーニング再開用のモデルパス")
    parser.add_argument("--lr", type=float, default=0.001, help="学習率")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger.info("トレーニングスクリプトのテスト実行") 