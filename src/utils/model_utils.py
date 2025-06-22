"""
モデル関連のユーティリティ関数
"""

import jax
import jax.numpy as jnp
import logging

# ロギング設定
logger = logging.getLogger(__name__)


class PolicyGradientLoss:
    """
    方策勾配損失関数
    
    方策勾配法（REINFORCE、A2Cなど）で使用する損失関数を集約したクラスです。
    主な損失関数は以下の通りです：
    
    - policy_loss: 方策に関する損失（アドバンテージ重み付けオプション）
    - value_loss: 価値推定に関する損失（二乗誤差）
    - entropy_loss: 方策のエントロピーに関する損失（探索を促進）
    
    また、複数の損失を組み合わせて計算する総合関数も提供します。
    """
    
    @staticmethod
    def policy_loss(logits, action_probs, advantages=None):
        """
        方策損失を計算する
        
        Args:
            logits: モデルの出力ロジット
            action_probs: ターゲット行動確率
            advantages: アドバンテージ値（Noneの場合は通常の交差エントロピー損失として計算）
            
        Returns:
            方策損失値
        """
        if advantages is None:
            # 通常の交差エントロピー損失
            return -jnp.sum(action_probs * jax.nn.log_softmax(logits)) / action_probs.shape[0]
        
        # アドバンテージによる重み付け交差エントロピー損失
        return -jnp.sum(action_probs * jax.nn.log_softmax(logits) * advantages) / logits.shape[0]
    
    @staticmethod
    @jax.jit
    def policy_gradient_loss(params, model_apply_fn, inputs, targets, feature_vector, advantages=None):
        """
        方策勾配損失関数（JIT最適化）
        
        モデルの適用と損失計算を一度に行うJIT最適化された関数です。
        
        Args:
            params: モデルパラメータ
            model_apply_fn: モデル適用関数
            inputs: 入力データ
            targets: ターゲット確率分布
            feature_vector: 特徴ベクトル
            advantages: アドバンテージ値（オプション）
            
        Returns:
            方策勾配損失値
        """
        # モデル推論
        policy_logits, _ = model_apply_fn(params, inputs, feature_vector=feature_vector)
        
        # 損失計算
        return PolicyGradientLoss.policy_loss(policy_logits, targets, advantages)
    
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
        return jnp.mean(jnp.square(values - target_values))
    
    @staticmethod
    def entropy_loss(logits):
        """
        エントロピー損失を計算する
        
        エントロピーが高いほど方策は均等に分布し、探索的になります。
        負の値を返すのは、最適化の際にエントロピーを最大化（損失を最小化）するためです。
        
        Args:
            logits: モデルの出力ロジット
            
        Returns:
            エントロピー損失値
        """
        # エントロピーの計算
        probabilities = jax.nn.softmax(logits)
        log_probabilities = jax.nn.log_softmax(logits)
        entropy = -jnp.sum(probabilities * log_probabilities, axis=1)
        return -jnp.mean(entropy)  # 最大化したいので負の値に
    
    @staticmethod
    def compute_losses_from_model_outputs(policy_logits, values, action_onehot, advantages, target_values, entropy_coeff=0.01):
        """
        モデル出力から損失を計算する総合関数
        
        個別の損失を計算し、それらを重み付けして総合損失を計算します。
        
        Args:
            policy_logits: モデルの方策出力
            values: モデルの価値出力
            action_onehot: ターゲット行動のone-hotエンコーディング
            advantages: アドバンテージ値
            target_values: ターゲット価値
            entropy_coeff: エントロピー係数（探索と活用のバランスを調整）
            
        Returns:
            (total_loss, (policy_loss, value_loss, entropy_loss)): 損失の値のタプル
        """
        # 各損失の計算
        policy_loss = PolicyGradientLoss.policy_loss(policy_logits, action_onehot, advantages)
        value_loss = PolicyGradientLoss.value_loss(values, target_values)
        entropy_loss = PolicyGradientLoss.entropy_loss(policy_logits)
        
        # 総合損失の計算
        total_loss = policy_loss + value_loss + entropy_coeff * entropy_loss
            
        return total_loss, (policy_loss, value_loss, entropy_loss)




