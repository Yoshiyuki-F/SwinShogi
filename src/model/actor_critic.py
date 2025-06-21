"""
方策と価値を予測するActorCriticネットワーク
"""

import jax
import jax.numpy as jnp
import logging
from typing import Dict, Tuple, Any, Optional
import functools

# ロギング設定
logger = logging.getLogger(__name__)

class ActorCritic:
    """
    ActorCriticネットワーク
    
    方策（行動確率）と価値（状態の価値）を予測するニューラルネットワーク
    SwinTransformerモデルをラップし、MCTSと連携する
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
        
        # バッチ処理用の関数
        self.batch_predict_jit = jax.jit(self._batch_predict)
    
    def predict(self, state_features: jnp.ndarray) -> Tuple[Dict[int, float], float]:
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
        # 動作高速化のため、確率が閾値以上の行動のみを返す
        action_probs = {i: float(prob) for i, prob in enumerate(policy_probs) if prob > 0.001}
        
        return action_probs, float(value)
    
    def batch_predict(self, states_batch: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        複数の状態を一括で予測する

        Args:
            states_batch: 状態の特徴量のバッチ [batch_size, height, width, channels]
            
        Returns:
            (方策ロジットのバッチ, 価値のバッチ)
        """
        return self.batch_predict_jit(self.params, states_batch)
    
    def get_features_from_transformer(self, state_features: jnp.ndarray) -> jnp.ndarray:
        """
        SwinTransformerからの特徴表現を抽出する
        シーケンス図のTransformer->AC部分に対応

        Args:
            state_features: 状態の特徴量
            
        Returns:
            特徴表現
        """
        # 実際の実装では、モデルから特徴抽出部分を呼び出す
        # ここではダミー実装
        if len(state_features.shape) == 3:
            state_features = jnp.expand_dims(state_features, axis=0)
            
        # この関数はモデルの中間層の出力を取得する目的
        # 実際の実装ではモデル側に特徴抽出関数を追加する必要がある
        features = self.model.apply(
            self.params, 
            state_features, 
            method=self.model.extract_features,
            deterministic=True
        )
        
        return features
    
    def _predict(self, params, state_features):
        """
        JIT最適化用の内部予測関数
        """
        return self.model.apply(params, state_features, deterministic=True)
    
    def _batch_predict(self, params, states_batch):
        """
        JIT最適化用のバッチ予測関数
        """
        return self.model.apply(params, states_batch, deterministic=True)
    
    def update(self, new_params):
        """
        パラメータを更新する
        
        Args:
            new_params: 新しいパラメータ
        """
        self.params = new_params

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_action_logits_and_value(self, params, state_features):
        """
        状態から行動ロジットと価値を計算する（JIT最適化版）
        
        Args:
            params: モデルパラメータ
            state_features: 状態特徴量
            
        Returns:
            (行動ロジット, 価値)
        """
        policy_logits, value = self.model.apply(params, state_features, deterministic=True)
        return policy_logits, value 