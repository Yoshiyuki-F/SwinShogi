"""
方策と価値を予測するActorCriticネットワーク
"""

import jax
import jax.numpy as jnp
import logging

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