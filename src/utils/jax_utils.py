"""
JAX関連のユーティリティ関数
"""

import jax
import jax.numpy as jnp
import time
import logging

# ロギング設定
logger = logging.getLogger(__name__)

def setup_jax(require_gpu=True):
    """
    JAXの設定を行う
    
    Args:
        require_gpu: GPUが必要かどうか
        
    Returns:
        設定が成功したかどうか
    """
    try:
        # GPUが利用可能かチェック
        jax.config.update('jax_enable_x64', True)  # 64ビット精度を有効化
        
        # デバイス情報表示
        devices = jax.devices()
        backend = jax.default_backend()
        logger.info(f"JAX認識デバイス: {devices}")
        logger.info(f"JAXデフォルトバックエンド: {backend}")
        
        # CPUが使用されている場合はエラーを発生させる
        if require_gpu and (backend == "cpu" or all("cpu" in str(d).lower() for d in devices)):
            raise ValueError("CPUが使用されています。GPUが必要です。")
        
        # GPUの使用状況を確認
        logger.info("GPU使用状況の確認:")
        
        # 小さな行列積のテスト
        x = jnp.ones((1000, 1000))
        start_time = time.time()
        for _ in range(10):
            y = jnp.dot(x, x)  # 行列積でGPUに負荷をかける
        jax.block_until_ready(y)
        end_time = time.time()
        logger.info(f"小さな行列積10回の実行時間: {(end_time - start_time)*1000:.2f}ms")
        
        # 大きな行列積のテスト（GPUメモリを多く使用）
        logger.info("GPUメモリ使用テスト:")
        try:
            # 5000x5000の行列を作成（約200MB）
            large_x = jnp.ones((5000, 5000), dtype=jnp.float32)
            start_time = time.time()
            large_y = jnp.dot(large_x, large_x)  # 約200MBのGPUメモリを使用
            jax.block_until_ready(large_y)
            end_time = time.time()
            logger.info(f"大きな行列積の実行時間: {(end_time - start_time)*1000:.2f}ms")
            
            # さらに大きな行列積のテスト（約800MB）
            larger_x = jnp.ones((10000, 10000), dtype=jnp.float32)
            start_time = time.time()
            larger_y = jnp.dot(larger_x, larger_x)
            jax.block_until_ready(larger_y)
            end_time = time.time()
            logger.info(f"より大きな行列積の実行時間: {(end_time - start_time)*1000:.2f}ms")
            logger.info("GPUメモリテスト成功！")
        except Exception as e:
            logger.warning(f"GPUメモリテスト中にエラーが発生しました: {e}")
            if require_gpu:
                return False
        
        logger.info("JAX設定が正常に完了しました。")
        return True
        
    except Exception as e:
        logger.error(f"JAX設定エラー: {e}")
        if require_gpu:
            logger.error("GPUが必要です。CUDA対応のJAXlibをインストールしてください。")
            return False
        else:
            logger.warning("GPUが検出されませんでしたが、CPUモードで続行します。")
            return True

def create_rng_keys(seed=0, num_keys=1):
    """
    複数のRNGキーを生成する
    
    Args:
        seed: 乱数シード
        num_keys: 生成するキーの数
        
    Returns:
        RNGキーのリスト
    """
    key = jax.random.PRNGKey(seed)
    if num_keys == 1:
        return key
    
    keys = []
    for _ in range(num_keys):
        key, subkey = jax.random.split(key)
        keys.append(subkey)
    
    return keys 