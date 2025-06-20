"""
パフォーマンス評価のためのユーティリティ関数

このモジュールでは、SwinShogiモデルのパフォーマンス評価とベンチマーク用の関数を提供します。
JAXの最適化機能を使用して推論速度を測定し、最適なバッチサイズを見つけるための関数が含まれています。
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional
import cProfile
import pstats
import io

def benchmark_inference(model, params, batch_sizes=[1, 2, 4, 8, 16, 32], n_runs=10, warm_up=3):
    """
    さまざまなバッチサイズでモデル推論のベンチマークを実行する
    
    Args:
        model: ベンチマークするモデル
        params: モデルのパラメータ
        batch_sizes: テストするバッチサイズのリスト
        n_runs: 各バッチサイズで実行する回数
        warm_up: ウォームアップ実行の回数（結果に含めない）
        
    Returns:
        バッチサイズごとの平均実行時間（ミリ秒）の辞書
    """
    results = {}
    
    # JIT最適化された推論関数
    @jax.jit
    def jitted_inference(params, inputs):
        return model.apply(params, inputs)
    
    for batch_size in batch_sizes:
        # テスト入力の作成
        inputs = jnp.ones((batch_size, 9, 9, 119))
        
        # ウォームアップ実行（コンパイル時間を除外）
        for _ in range(warm_up):
            _ = jitted_inference(params, inputs)
            
        # 計測実行
        times = []
        for _ in range(n_runs):
            start_time = time.time()
            _ = jitted_inference(params, inputs)
            jax.block_until_ready(_)  # 計算が完了するまで待機
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ミリ秒に変換
        
        results[batch_size] = np.mean(times)
    
    return results

def find_optimal_batch_size(model, params, sample_shape=(9, 9, 119), min_batch=1, max_batch=64, step=2):
    """
    スループットを最大化する最適なバッチサイズを見つける
    
    Args:
        model: テストするモデル
        params: モデルのパラメータ
        sample_shape: 単一サンプルの形状
        min_batch: テストする最小バッチサイズ
        max_batch: テストする最大バッチサイズ
        step: バッチサイズの増分
        
    Returns:
        (最適なバッチサイズ, サンプルごとの最小処理時間)
    """
    @jax.jit
    def jitted_inference(params, inputs):
        return model.apply(params, inputs)
    
    # ウォームアップ実行
    _ = jitted_inference(params, jnp.ones((1,) + sample_shape))
    
    best_batch_size = 1
    best_throughput = 0
    
    for batch_size in range(min_batch, max_batch + 1, step):
        try:
            inputs = jnp.ones((batch_size,) + sample_shape)
            
            # 実行時間を測定
            start_time = time.time()
            _ = jitted_inference(params, inputs)
            jax.block_until_ready(_)
            end_time = time.time()
            
            execution_time = end_time - start_time
            throughput = batch_size / execution_time
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
                
            print(f"Batch size: {batch_size}, Time: {execution_time*1000:.2f}ms, "
                  f"Throughput: {throughput:.2f} samples/s")
                  
        except (RuntimeError, jax.errors.OutOfMemoryError) as e:
            print(f"Error at batch size {batch_size}: {e}")
            break
    
    return best_batch_size, 1.0 / best_throughput

def profile_mcts(mcts, n_simulations=100, include_primitives=False):
    """
    MCTSの探索処理をプロファイリングする
    
    Args:
        mcts: プロファイリングするMCTSインスタンス
        n_simulations: 実行するシミュレーション回数
        include_primitives: プリミティブ関数（小さな関数）を結果に含めるかどうか
        
    Returns:
        関数ごとの実行時間（ミリ秒）の辞書
    """
    # プロファイラを設定
    pr = cProfile.Profile()
    pr.enable()
    
    # シミュレーションを実行
    for _ in range(n_simulations):
        mcts.search()
    
    pr.disable()
    
    # 結果を文字列バッファに出力
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    
    if not include_primitives:
        # プリミティブ関数（実行回数が多い小さな関数）を除外
        ps.print_stats(0.1)
    else:
        ps.print_stats()
    
    # 結果を解析して辞書に変換
    result_dict = {}
    lines = s.getvalue().strip().split('\n')
    
    for line in lines[5:]:  # ヘッダー行をスキップ
        if not line or line.startswith(' '):
            continue
            
        parts = line.strip().split()
        if len(parts) >= 6:
            cumtime = float(parts[3])
            func_name = ' '.join(parts[5:])
            
            # モジュール名と関数名を抽出
            if ':' in func_name:
                func_name = func_name.split(':')[1]
                
            # 括弧内の情報を除去
            if '(' in func_name:
                func_name = func_name.split('(')[0]
            
            result_dict[func_name] = cumtime * 1000  # ミリ秒に変換
    
    return result_dict

def compare_devices(model, params, input_shape=(1, 9, 9, 119), n_runs=10):
    """
    異なるデバイス（CPU vs GPU）でモデルのパフォーマンスを比較する
    
    Args:
        model: テストするモデル
        params: モデルのパラメータ
        input_shape: 入力テンソルの形状
        n_runs: 各デバイスで実行する回数
        
    Returns:
        各デバイスの平均実行時間（ミリ秒）の辞書
    """
    results = {}
    
    # 利用可能なデバイスを取得
    devices = jax.devices()
    cpu_devices = [d for d in devices if d.platform == 'cpu']
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    
    # CPUでテスト
    if cpu_devices:
        with jax.devices(cpu_devices):
            # JIT最適化された推論関数
            @jax.jit
            def cpu_inference(params, inputs):
                return model.apply(params, inputs)
            
            # テスト入力
            inputs = jnp.ones(input_shape)
            
            # ウォームアップ
            _ = cpu_inference(params, inputs)
            
            # 計測実行
            times = []
            for _ in range(n_runs):
                start_time = time.time()
                output = cpu_inference(params, inputs)
                jax.block_until_ready(output)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
            results['CPU'] = np.mean(times)
    
    # GPUでテスト
    if gpu_devices:
        with jax.devices(gpu_devices):
            # JIT最適化された推論関数
            @jax.jit
            def gpu_inference(params, inputs):
                return model.apply(params, inputs)
            
            # テスト入力
            inputs = jnp.ones(input_shape)
            
            # ウォームアップ
            _ = gpu_inference(params, inputs)
            
            # 計測実行
            times = []
            for _ in range(n_runs):
                start_time = time.time()
                output = gpu_inference(params, inputs)
                jax.block_until_ready(output)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
            results['GPU'] = np.mean(times)
    
    return results

def measure_model_size(params):
    """
    モデルのパラメータサイズをメガバイト単位で推定する
    
    Args:
        params: モデルのパラメータ
        
    Returns:
        パラメータのサイズ（MB）
    """
    # パラメータのフラットなリストを取得
    param_leaves = jax.tree_leaves(params)
    
    # 各パラメータの要素数を計算
    param_counts = [np.prod(p.shape) for p in param_leaves]
    
    # パラメータの総数
    total_params = sum(param_counts)
    
    # パラメータタイプを取得（通常はfloat32）
    dtype = param_leaves[0].dtype
    bytes_per_param = np.dtype(dtype).itemsize
    
    # サイズをメガバイトで計算
    size_bytes = total_params * bytes_per_param
    size_mb = size_bytes / (1024 * 1024)
    
    return {
        'parameter_count': total_params,
        'dtype': str(dtype),
        'size_mb': size_mb
    }

def get_device_memory_info():
    """
    JAXデバイスのメモリ使用状況を取得する
    
    Returns:
        デバイスのメモリ情報の辞書（デバイスタイプによって異なる）
    """
    devices = jax.devices()
    result = {}
    
    for i, device in enumerate(devices):
        device_type = device.platform
        device_info = {
            'platform': device_type,
            'device_id': i,
        }
        
        # GPU固有のメモリ情報を取得（JAXがXLAを介してアクセスできる場合）
        try:
            # メモリ情報の取得を試みる（GPU専用）
            if device_type == 'gpu':
                # この関数はJAXの実装に依存するため、将来変更される可能性がある
                memory_info = jax.devices()[i].memory_stats()
                device_info.update(memory_info)
        except (AttributeError, NotImplementedError):
            # メモリ情報を取得できない場合は空の辞書を返す
            pass
            
        result[f"{device_type}_{i}"] = device_info
    
    return result