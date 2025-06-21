"""
モデル関連のユーティリティ関数
"""

import jax
import jax.numpy as jnp
import logging
from src.shogi.board_encoder import encode_board_state, get_feature_vector

# ロギング設定
logger = logging.getLogger(__name__)

# JIT最適化された予測関数
@jax.jit
def predict_jit(model, params, state_features, feature_vector=None):
    """
    JAXのJIT最適化された推論関数
    
    この関数はJITコンパイルされているため、繰り返し呼び出されると大幅な速度向上が得られます。
    一見すると単純なラッパーのように見えますが、JIT最適化によるパフォーマンス向上が目的です。
    
    Args:
        model: SwinShogiモデルまたはActorCriticモデル
        params: モデルパラメータ
        state_features: 状態の特徴量
        feature_vector: 追加の特徴ベクトル（オプション）
        
    Returns:
        (方策ロジット, 価値)
    """
    if feature_vector is not None:
        return model.apply(params, state_features, feature_vector=feature_vector, deterministic=True)
    else:
        return model.apply(params, state_features, deterministic=True)

def predict(model, params, state=None, state_features=None):
    """
    モデル推論を実行する。将棋の状態またはエンコード済みの特徴量から方策と価値を予測する。
    
    Args:
        model: SwinShogiモデルまたはActorCriticモデル
        params: モデルパラメータ
        state: 将棋の状態（辞書形式）。state_featuresが指定されている場合は無視される。
        state_features: 事前にエンコードされた状態の特徴量。Noneの場合はstateからエンコードする。
            
    Returns:
        方策（辞書形式）と価値のタプル、または生のロジットと値
    """
    # 状態特徴量の準備
    if state_features is None and state is not None:
        # 状態を符号化
        board_encoded = encode_board_state(state)
        state_features = jnp.array(board_encoded).reshape(1, 9, 9, 2)
        
        # 特徴ベクトルを取得（state_featuresに直接含めることもできる）
        feature_vector = get_feature_vector(state)
        feature_tensor = jnp.array(feature_vector).reshape(1, -1)
        
        # モデル推論
        policy_logits, value = predict_jit(model, params, state_features, feature_tensor)
    else:
        # 特徴量が直接与えられている場合
        # バッチ次元の確認
        if len(state_features.shape) == 3:
            state_features = jnp.expand_dims(state_features, axis=0)
            
        # モデル推論
        if hasattr(model, 'apply'):
            # Flaxモデルの場合
            policy_logits, value = predict_jit(model, params, state_features)
        else:
            # ActorCriticインスタンス自体が渡された場合（ただし非推奨）
            policy_logits, value = model.predict_jit(params, state_features)
            
    # バッチ次元が1の場合は削除
    if policy_logits.shape[0] == 1:
        policy_logits = policy_logits[0]
        value = value[0][0] if len(value.shape) > 1 else value[0]
        
        # 方策ロジットを確率に変換
        policy_probs = jax.nn.softmax(policy_logits)
        
        # 確率を辞書形式に変換（閾値以上の行動のみ）
        action_probs = {i: float(prob) for i, prob in enumerate(policy_probs) if prob > 0.001}
        
        return action_probs, float(value)
    else:
        # バッチ処理の場合はロジットと値をそのまま返す
        return policy_logits, value

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




