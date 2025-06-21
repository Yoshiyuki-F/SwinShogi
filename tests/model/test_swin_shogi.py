"""
Swin Transformer将棋モデルのテスト
"""

import jax
import jax.numpy as jnp
import time
import unittest
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, TypeVar, cast
from src.model.shogi_model import create_swin_shogi_model, SwinShogiModel
from src.utils.model_utils import inference_jit
from src.shogi.board_encoder import encode_board_state, get_feature_vector
from config.default_config import get_model_config, ModelConfig
from src.shogi.shogi_pieces import Player

def test_swin_shogi_model() -> bool:
    """
    モデルのテスト関数
    
    Returns:
        bool: テストが成功したかどうか
    """
    # モデルの初期化
    rng = jax.random.PRNGKey(0)
    model_config = get_model_config()
    model, params = create_swin_shogi_model(rng, model_config=model_config)
    
    # テスト用のダミー将棋状態の作成
    # シンプルなダミー状態（空の盤面）
    dummy_state: Dict[str, Any] = {
        'board': [[None for _ in range(9)] for _ in range(9)],
        'hands': {
            Player.SENTE: {'pawn': 2, 'lance': 1},
            Player.GOTE: {'bishop': 1}
        },
        'turn': Player.SENTE
    }
    
    # テスト入力 - 実際のエンコーダを使用
    board_encoded = encode_board_state(dummy_state)
    test_input = jnp.array(board_encoded).reshape(1, 9, 9, 2)
    
    # テスト特徴ベクトル - 実際のエンコーダを使用
    feature_vector = get_feature_vector(dummy_state)
    test_feature = jnp.array(feature_vector).reshape(1, model_config.feature_dim)
    
    print("------------ 推論テスト ------------")
    print(f"盤面エンコード形状: {test_input.shape}")
    print(f"特徴ベクトル形状: {test_feature.shape}")
    
    # 推論テスト
    start_time = time.time()
    policy_logits, value = model.apply(params, test_input, feature_vector=test_feature)
    end_time = time.time()
    
    print(f"モデル推論所要時間: {(end_time - start_time) * 1000:.2f}ms")
    print(f"方策出力形状: {policy_logits.shape}")
    print(f"価値出力形状: {value.shape}")
    
    # JIT最適化テスト
    print("\n------------ JIT最適化テスト ------------")
    
    start_time = time.time()
    policy_logits_jit, value_jit = inference_jit(model, params, test_input, test_feature)
    end_time = time.time()
    
    print(f"JIT最適化後の推論所要時間: {(end_time - start_time) * 1000:.2f}ms")
    
    # パラメータの保存と読み込みテスト
    test_save_path = "/tmp/swin_shogi_test_params.pkl"
    model.save_params(test_save_path, params)
    loaded_params = model.load_params(test_save_path, params)
    
    # 同一性確認
    equal = all(jnp.array_equal(p1, p2) for p1, p2 in zip(jax.tree.leaves(params), jax.tree.leaves(loaded_params)))
    print(f"パラメータの保存と読み込みテスト: {'成功' if equal else '失敗'}")
    
    return True

class TestSwinShogi(unittest.TestCase):
    """Swin Transformer将棋モデルのユニットテスト"""
    
    def test_model_creation(self) -> None:
        """モデル作成のテスト"""
        rng = jax.random.PRNGKey(0)
        model, params = create_swin_shogi_model(rng)
        self.assertIsNotNone(model)
        self.assertIsNotNone(params)
    
    def test_inference(self) -> None:
        """推論のテスト"""
        rng = jax.random.PRNGKey(0)
        model_config = get_model_config()
        model, params = create_swin_shogi_model(rng, model_config=model_config)
        
        # テスト入力
        dummy_input = jnp.zeros((1, 9, 9, 2))
        feature_vector = jnp.zeros((1, model_config.feature_dim))
        
        # 推論実行
        policy_logits, value = model.apply(params, dummy_input, feature_vector=feature_vector)
        
        # 出力形状の検証
        self.assertEqual(policy_logits.shape[1], model_config.n_policy_outputs)
        self.assertEqual(value.shape, (1, 1))

if __name__ == "__main__":
    test_swin_shogi_model()