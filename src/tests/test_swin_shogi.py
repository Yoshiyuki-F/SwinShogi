"""
SwinShogiモデルのテストスクリプト
"""
import unittest
import os
import tempfile
import jax
import jax.numpy as jnp
from src.model.swin_shogi import SwinShogiModel, create_swin_shogi_model

class TestSwinShogiModel(unittest.TestCase):
    """SwinShogiモデルのテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.rng = jax.random.PRNGKey(42)
        self.model, self.params = create_swin_shogi_model(self.rng)
        self.test_input = jnp.ones((1, 9, 9, 119))  # バッチサイズ1のテスト入力
        
    def test_model_initialization(self):
        """モデル初期化のテスト"""
        # モデルのインスタンスが正しく作成されることを確認
        self.assertIsInstance(self.model, SwinShogiModel)
        
        # パラメータが初期化されることを確認
        self.assertIsNotNone(self.params)
        
        # 各レイヤーが正しく初期化されていることを確認
        self.assertEqual(len(self.model.depths), len(self.model.num_heads))
        self.assertEqual(len(self.model.layers), len(self.model.depths))
        
    def test_forward_pass(self):
        """順伝播のテスト"""
        # 順伝播が正常に動作することを確認
        policy_logits, value = self.model.apply(self.params, self.test_input)
        
        # 出力の形状を確認
        self.assertEqual(policy_logits.shape, (1, self.model.n_policy_outputs))
        self.assertEqual(value.shape, (1, 1))
        
        # 政策の出力が適切な範囲であることを確認
        policy_probs = jax.nn.softmax(policy_logits)
        self.assertTrue(jnp.all(policy_probs >= 0))
        self.assertTrue(jnp.all(policy_probs <= 1))
        self.assertAlmostEqual(float(jnp.sum(policy_probs)), 1.0, places=5)
        
        # 価値の出力が適切な範囲であることを確認
        self.assertTrue(jnp.all(value >= -1))
        self.assertTrue(jnp.all(value <= 1))
        
    def test_save_load_params(self):
        """パラメータの保存と読み込みのテスト"""
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # パラメータを保存
            self.model.save_params(temp_path, self.params)
            
            # ファイルが作成されたことを確認
            self.assertTrue(os.path.exists(temp_path))
            
            # パラメータを読み込み
            loaded_params = self.model.load_params(temp_path)
            
            # 読み込んだパラメータが元のパラメータと一致することを確認
            for p1, p2 in zip(jax.tree_leaves(self.params), 
                             jax.tree_leaves(loaded_params)):
                self.assertTrue(jnp.array_equal(p1, p2))
                
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_cross_entropy_loss(self):
        """交差エントロピー損失関数のテスト"""
        # テスト用のデータを作成
        batch_size = 2
        logits = jnp.array([[1.0, 2.0, 0.5], [0.7, 1.3, 2.1]])
        targets = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        
        # 損失関数を計算
        loss = SwinShogiModel.cross_entropy_loss(logits, targets)
        
        # 損失が計算されることを確認
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)  # 損失は正の値
        
        # 自明な入力での損失をチェック
        uniform_targets = jnp.ones_like(logits) / logits.shape[1]
        uniform_loss = SwinShogiModel.cross_entropy_loss(jnp.zeros_like(logits), uniform_targets)
        self.assertAlmostEqual(float(uniform_loss), jnp.log(logits.shape[1]), places=5)
        
    def test_batch_inference(self):
        """バッチ処理のテスト"""
        # バッチサイズを増やしたテスト入力
        batch_size = 4
        batch_input = jnp.ones((batch_size, 9, 9, 119))
        
        # バッチ処理での推論
        policy_logits, value = self.model.apply(self.params, batch_input)
        
        # 出力の形状を確認
        self.assertEqual(policy_logits.shape, (batch_size, self.model.n_policy_outputs))
        self.assertEqual(value.shape, (batch_size, 1))
        
    def test_jit_compilation(self):
        """JITコンパイルのテスト"""
        # JIT最適化した関数を定義
        @jax.jit
        def jitted_forward(params, x):
            return self.model.apply(params, x)
        
        # JIT関数が動作することを確認
        policy_logits, value = jitted_forward(self.params, self.test_input)
        
        # 出力の形状を確認
        self.assertEqual(policy_logits.shape, (1, self.model.n_policy_outputs))
        self.assertEqual(value.shape, (1, 1))

if __name__ == "__main__":
    unittest.main()