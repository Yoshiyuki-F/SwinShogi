"""
方策と価値を予測するActorCriticネットワーク
"""

import jax.numpy as jnp
import logging

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
            params: モデルパラメータ
        """
        self.model = model
        self.params = params
    
    def get_features_from_transformer(self, state_features: jnp.ndarray) -> jnp.ndarray:
        """
        SwinTransformerからの特徴表現を抽出する
        シーケンス図のTransformer->AC部分に対応

        Args:
            state_features: 状態の特徴量
            
        Returns:
            特徴表現
        """
        # バッチ次元の確認
        if len(state_features.shape) == 3:
            state_features = jnp.expand_dims(state_features, axis=0)
            
        # モデルの特徴抽出メソッドを呼び出す
        features = self.model.apply(
            self.params, 
            state_features, 
            method=self.model.extract_features,
            deterministic=True
        )
        
        return features
    
    def update(self, new_params):
        """
        パラメータを更新する
        
        Args:
            new_params: 新しいパラメータ
        """
        self.params = new_params 