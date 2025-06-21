"""
SwinShogiのパフォーマンス評価テスト

このモジュールでは、SwinShogiの推論速度や探索効率を評価するためのテストを提供します。
"""
import unittest
import time
from typing import Dict, List, Any, Optional, Tuple, Union, cast
import jax
import jax.numpy as jnp
from src.model.swin_shogi import create_swin_shogi_model
from src.utils.performance import benchmark_inference, profile_mcts
from src.rl.mcts import MCTS
from src.model.actor_critic import ActorCritic
from src.shogi.shogi_game import ShogiGame

class TestPerformanceEvaluation(unittest.TestCase):
    """パフォーマンス評価のテストケース"""

    def setUp(self) -> None:
        """テスト前の準備"""
        # モデルの初期化
        self.rng = jax.random.PRNGKey(42)
        self.model, self.params = create_swin_shogi_model(self.rng)
        
        # テスト用データの作成
        self.test_input_single = jnp.ones((1, 9, 9, 119))
        self.test_input_batch = jnp.ones((8, 9, 9, 119))

    def test_inference_speed(self) -> None:
        """推論速度のテスト"""
        print("\n===== 推論速度テスト =====")
        
        # 単一サンプルでの推論速度テスト
        print("単一サンプルの推論テスト:")
        
        # 標準推論
        start_time = time.time()
        policy_logits, value = self.model.apply(self.params, self.test_input_single)
        standard_inference_time = time.time() - start_time
        print(f"標準推論時間: {standard_inference_time*1000:.2f}ms")
        
        # JIT最適化推論
        jit_inference_fn = jax.jit(lambda p, x: self.model.apply(p, x))
        
        # コンパイル時間を含む最初の実行
        start_time = time.time()
        policy_jit, value_jit = jit_inference_fn(self.params, self.test_input_single)
        first_jit_time = time.time() - start_time
        print(f"JIT初回推論時間（コンパイル含む）: {first_jit_time*1000:.2f}ms")
        
        # 2回目の実行（コンパイル済み）
        start_time = time.time()
        policy_jit, value_jit = jit_inference_fn(self.params, self.test_input_single)
        jit_time = time.time() - start_time
        print(f"JIT2回目推論時間: {jit_time*1000:.2f}ms")
        print(f"JIT高速化率: {standard_inference_time/jit_time:.2f}倍")
        
        # バッチ処理テスト
        print("\nバッチ処理テスト:")
        batch_size = self.test_input_batch.shape[0]
        
        # 1サンプルずつ処理
        start_time = time.time()
        for i in range(batch_size):
            sample = self.test_input_batch[i:i+1]
            policy, value = jit_inference_fn(self.params, sample)
        individual_time = time.time() - start_time
        print(f"{batch_size}サンプルを個別に推論: {individual_time*1000:.2f}ms")
        
        # バッチ処理
        start_time = time.time()
        policy_batch, value_batch = jit_inference_fn(self.params, self.test_input_batch)
        batch_time = time.time() - start_time
        print(f"{batch_size}サンプルをバッチ推論: {batch_time*1000:.2f}ms")
        print(f"バッチ高速化率: {individual_time/batch_time:.2f}倍")
        
        # ベンチマーク関数を使用
        results = benchmark_inference(self.model, self.params, batch_sizes=[1, 2, 4, 8, 16])
        print("\n推論ベンチマーク結果:")
        for batch_size, time_ms in results.items():
            print(f"バッチサイズ {batch_size}: {time_ms:.2f}ms")
            
        # 結果の検証
        self.assertLess(jit_time, standard_inference_time)
        self.assertLess(batch_time, individual_time)

    def test_mcts_performance(self) -> None:
        """MCTSのパフォーマンステスト"""
        print("\n===== MCTS性能テスト =====")
        
        # ゲームとMCTSの初期化
        game = ShogiGame()
        actor_critic = ActorCritic(self.model, self.params)
        mcts = MCTS(game, actor_critic)
        
        # MCTSの実行時間を計測
        n_simulations = 100
        start_time = time.time()
        for _ in range(n_simulations):
            mcts.search()
        total_time = time.time() - start_time
        
        print(f"{n_simulations}回のMCTS探索時間: {total_time*1000:.2f}ms")
        print(f"1探索あたりの平均時間: {total_time*1000/n_simulations:.2f}ms")
        
        # プロファイリング結果
        profile_results = profile_mcts(mcts, n_simulations=10)
        print("\nMCTSプロファイリング結果:")
        for func_name, time_ms in profile_results.items():
            print(f"{func_name}: {time_ms:.2f}ms")
            
        # 探索ノード数のテスト
        self.assertGreater(mcts.root.visit_count, n_simulations)  # ルートノードの訪問回数

class TestJAXOptimizations(unittest.TestCase):
    """JAXの最適化に関するテスト"""

    def test_device_placement(self) -> None:
        """JAXのデバイス配置テスト"""
        print("\n===== JAXデバイス配置テスト =====")
        
        devices = jax.devices()
        print(f"利用可能なデバイス: {devices}")
        
        # CPUが利用可能かどうか
        cpu_devices = [d for d in devices if d.platform == 'cpu']
        if cpu_devices:
            print(f"CPU利用可能: {cpu_devices}")
            
        # GPUが利用可能かどうか
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if gpu_devices:
            print(f"GPU利用可能: {gpu_devices}")
        else:
            print("GPU利用不可")
            
        # TPUが利用可能かどうか
        tpu_devices = [d for d in devices if d.platform == 'tpu']
        if tpu_devices:
            print(f"TPU利用可能: {tpu_devices}")
        else:
            print("TPU利用不可")
            
        # JAX設定の表示
        print(f"JAX設定:")
        print(f"- デフォルトバックエンド: {jax.default_backend()}")
        print(f"- 64bit精度有効: {jax.config.jax_enable_x64}")
            
        # テスト用配列の作成（どのデバイスで計算されるか）
        x = jnp.ones((1000, 1000))
        y = jnp.ones((1000, 1000))
        
        start_time = time.time()
        z = x @ y  # 行列乗算
        elapsed = time.time() - start_time
        
        print(f"1000x1000行列乗算時間: {elapsed*1000:.2f}ms")
        
        # デバイス間転送テスト（GPUがある場合）
        if gpu_devices:
            cpu_array = jnp.ones((1000, 1000))
            start_time = time.time()
            gpu_array = jax.device_put(cpu_array, gpu_devices[0])
            transfer_time = time.time() - start_time
            print(f"CPU->GPU転送時間（1000x1000配列）: {transfer_time*1000:.2f}ms")

if __name__ == "__main__":
    unittest.main()