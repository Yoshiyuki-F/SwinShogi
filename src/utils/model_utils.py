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

def calculate_gradient_norm(grads):
    """
    勾配のグローバルノルムを計算する
    
    Args:
        grads: パラメータ勾配のtree構造
    
    Returns:
        float: 勾配のグローバルノルム
    """
    flat_grads = jax.tree_leaves(grads)
    return jnp.sqrt(jnp.sum(jnp.array([jnp.sum(jnp.square(g)) for g in flat_grads])))

def clip_gradients(grads, max_norm):
    """
    勾配のクリッピングを行う
    
    Args:
        grads: パラメータ勾配のtree構造
        max_norm: クリッピングの最大値
    
    Returns:
        クリッピングされた勾配のtree構造
    """
    return jax.tree_map(
        lambda g: jnp.clip(g, -max_norm, max_norm),
        grads
    )

def process_gradients(grads, max_norm):
    """
    勾配の計算とクリッピングを行う
    
    Args:
        grads: モデルの勾配
        max_norm: 勾配クリッピングの最大ノルム
    
    Returns:
        (clipped_grads, global_norm): クリッピングされた勾配とそのグローバルノルム
    """
    # 勾配のノルムを計算
    global_norm = calculate_gradient_norm(grads)
    
    # 勾配クリッピング
    clipped_grads = clip_gradients(grads, max_norm)
    
    return clipped_grads, global_norm

class PolicyGradientLoss:
    """
    方策勾配損失関数
    
    方策勾配法（REINFORCE、A2Cなど）で使用する損失関数を実装します。
    """
    
    @staticmethod
    def cross_entropy_loss(logits, targets):
        """
        交差エントロピー損失を計算する
        
        Args:
            logits: モデルの出力ロジット
            targets: ターゲット確率分布またはone-hotエンコーディング
            
        Returns:
            交差エントロピー損失値
        """
        return -jnp.sum(targets * jax.nn.log_softmax(logits)) / targets.shape[0]
    
    @staticmethod
    @jax.jit
    def policy_gradient_loss(params, model_apply_fn, inputs, targets, feature_vector):
        """
        方策勾配損失関数（JIT最適化）
        
        Args:
            params: モデルパラメータ
            model_apply_fn: モデル適用関数
            inputs: 入力データ
            targets: ターゲット確率分布
            feature_vector: 特徴ベクトル
            
        Returns:
            方策勾配損失値
        """
        policy_logits, _ = model_apply_fn(params, inputs, feature_vector=feature_vector)
        return PolicyGradientLoss.cross_entropy_loss(policy_logits, targets)
    
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
        # 交差エントロピー損失にアドバンテージを重み付け
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
        
    @staticmethod
    def compute_losses_from_model_outputs(policy_logits, values, action_onehot, advantages, target_values, entropy_coeff=0.01):
        """
        モデル出力から損失を計算する総合関数
        
        Args:
            policy_logits: モデルの方策出力
            values: モデルの価値出力
            action_onehot: ターゲット行動のone-hotエンコーディング
            advantages: アドバンテージ値
            target_values: ターゲット価値
            entropy_coeff: エントロピー係数
            
        Returns:
            (total_loss, (policy_loss, value_loss, entropy_loss)): 損失の値のタプル
        """
        # 損失の計算
        policy_loss = PolicyGradientLoss.policy_loss(policy_logits, action_onehot, advantages)
        value_loss = PolicyGradientLoss.value_loss(values, target_values)
        entropy_loss = PolicyGradientLoss.entropy_loss(policy_logits)
        total_loss = PolicyGradientLoss.total_loss(
            policy_loss, value_loss, entropy_loss, entropy_coeff)
            
        return total_loss, (policy_loss, value_loss, entropy_loss)

# 後方互換性のために個別の関数も提供
cross_entropy_loss = PolicyGradientLoss.cross_entropy_loss
policy_gradient_loss = PolicyGradientLoss.policy_gradient_loss



