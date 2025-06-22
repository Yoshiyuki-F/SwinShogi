"""
方策と価値を予測するActorCriticネットワーク
"""
import jax
import jax.numpy as jnp
import logging
from src.shogi.board_encoder import encode_board_state, get_feature_vector
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


    @staticmethod
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

    @staticmethod
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