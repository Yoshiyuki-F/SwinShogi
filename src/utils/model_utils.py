"""
モデル関連のユーティリティ関数
"""

import jax
import jax.numpy as jnp
import logging
from src.shogi.board_encoder import encode_board_state, get_feature_vector

# ロギング設定
logger = logging.getLogger(__name__)

def predict(model, params, state):
    """
    将棋の状態からモデル推論を実行する
    
    Args:
        model: SwinShogiモデル
        params: モデルパラメータ
        state: 将棋の状態（辞書形式）
        
    Returns:
        policy_logits, value: 方策ロジットと価値
    """
    # 状態を符号化
    board_encoded = encode_board_state(state)
    board_tensor = jnp.array(board_encoded).reshape(1, 9, 9, 2)
    
    # 特徴ベクトルを取得
    feature_vector = get_feature_vector(state)
    feature_tensor = jnp.array(feature_vector).reshape(1, -1)
    
    # モデル推論
    return model.apply(params, board_tensor, feature_vector=feature_tensor)

def inference_jit(model, params, inputs, feature_vector):
    """推論関数（JIT最適化）"""
    @jax.jit
    def _inference(params, inputs, feature_vector):
        return model.apply(params, inputs, feature_vector=feature_vector)
    
    return _inference(params, inputs, feature_vector)

def cross_entropy_loss(logits, targets):
    """交差エントロピー損失関数"""
    return -jnp.sum(targets * jax.nn.log_softmax(logits)) / targets.shape[0]

@jax.jit
def policy_gradient_loss(params, model_apply_fn, inputs, targets, feature_vector):
    """方策勾配損失関数（JIT最適化）"""
    policy_logits, _ = model_apply_fn(params, inputs, feature_vector=feature_vector)
    return cross_entropy_loss(policy_logits, targets)


