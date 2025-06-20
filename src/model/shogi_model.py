"""
将棋用のSwin Transformerモデル実装
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import flax
import time
from config.default_config import get_model_config, ModelConfig
from src.model.swin_transformer import (
    PatchEmbed, BasicLayer, PatchMerging
)

class SwinShogiModel(flax.linen.Module):
    """Swin Transformerベースの将棋モデル"""
    # 設定値をデフォルトのconfigから取得
    model_config: ModelConfig = None
    
    def setup(self):
        # configが指定されていない場合はデフォルト値を使用
        if self.model_config is None:
            self.model_config = get_model_config()
        
        config = self.model_config
        self.num_layers = len(config.depths)  # 2層
        # 1層目: 9x9 -> 2層目: 3x3（3で割るので）
        self.num_features = int(config.embed_dim * 9 ** (self.num_layers - 1))
        
        # パッチ分割と埋め込み
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim
        )
        
        # パッチ解像度を計算
        patches_resolution = (
            config.img_size[0] // config.patch_size[0],
            config.img_size[1] // config.patch_size[1]
        )
        
        # 絶対位置埋め込み
        self.absolute_pos_embed = self.param(
            'absolute_pos_embed',
            flax.linen.initializers.normal(0.02),
            (1, patches_resolution[0] * patches_resolution[1], config.embed_dim)
        )
        
        # ドロップアウト
        self.pos_drop = flax.linen.Dropout(rate=config.drop_rate)
        
        # 段階的なドロップパス率
        dpr = [x for x in np.linspace(0, config.drop_path_rate, sum(config.depths))]
        
        # 2層構造のネットワーク構築
        self.layers = {
            f'layer{i}': BasicLayer(
                dim=int(config.embed_dim * 9 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (3 ** i_layer),
                    patches_resolution[1] // (3 ** i_layer)
                ),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                norm_layer=config.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=config.use_checkpoint
            )
            for i_layer, i in enumerate(range(self.num_layers))
        }
        
        # 最終正規化層
        self.norm = config.norm_layer()
        
        # 政策ヘッド（移動確率）
        self.policy_head = flax.linen.Sequential([
            flax.linen.Dense(512),
            flax.linen.gelu,
            flax.linen.Dense(config.n_policy_outputs)
        ])
        
        # 価値ヘッド（勝率評価）
        self.value_head = flax.linen.Sequential([
            flax.linen.Dense(256),
            flax.linen.gelu,
            flax.linen.Dense(1),
            flax.linen.tanh
        ])
    
    def __call__(self, x, deterministic: bool = True):
        # パッチ埋め込み
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        
        # 位置埋め込みの追加
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x, deterministic=deterministic)
        
        # 現在の解像度を計算
        config = self.model_config
        current_resolution = (
            config.img_size[0] // config.patch_size[0],
            config.img_size[1] // config.patch_size[1]
        )
        
        # Swin Transformer段階
        for layer in self.layers.values():
            x, current_resolution = layer(x, current_resolution=current_resolution, deterministic=deterministic)
        
        # 最終正規化
        x = self.norm(x)
        x = jnp.mean(x, axis=1)  # グローバル平均プーリング
        
        # ヘッド
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
    
    def save_params(self, path: str, params):
        """モデルパラメータを保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(flax.serialization.to_bytes(params))
    
    def load_params(self, path: str, params):
        """モデルパラメータを読み込み"""
        with open(path, 'rb') as f:
            loaded_params = flax.serialization.from_bytes(params, f.read())
        return loaded_params
    
    @staticmethod
    def cross_entropy_loss(logits, targets):
        """交差エントロピー損失関数"""
        return -jnp.sum(targets * jax.nn.log_softmax(logits)) / targets.shape[0]
    
    @staticmethod
    @jax.jit
    def policy_gradient_loss(params, model_apply_fn, inputs, targets):
        """方策勾配損失関数（JIT最適化）"""
        policy_logits, _ = model_apply_fn(params, inputs)
        return SwinShogiModel.cross_entropy_loss(policy_logits, targets)

def create_swin_shogi_model(rng, model_config=None, batch_size=1):
    """SwinShogiモデルの作成"""
    if model_config is None:
        model_config = get_model_config()
    
    model = SwinShogiModel(model_config=model_config)
    input_shape = (batch_size, model_config.img_size[0], model_config.img_size[1], model_config.in_chans)
    params = model.init(rng, jnp.ones(input_shape))
    return model, params

def inference_jit(model, params, inputs):
    """推論関数（JIT最適化）"""
    @jax.jit
    def _inference(params, inputs):
        return model.apply(params, inputs)
    
    return _inference(params, inputs)

def test_swin_shogi_model():
    """モデルのテスト関数"""
    # モデルの初期化
    rng = jax.random.PRNGKey(0)
    model, params = create_swin_shogi_model(rng)
    
    # テスト入力
    test_input = jnp.ones((1, 9, 9, 119))
    
    # 推論テスト
    start_time = time.time()
    policy_logits, value = model.apply(params, test_input)
    end_time = time.time()
    
    print(f"モデル推論所要時間: {(end_time - start_time) * 1000:.2f}ms")
    print(f"方策出力形状: {policy_logits.shape}")
    print(f"価値出力形状: {value.shape}")
    
    # JIT最適化テスト
    start_time = time.time()
    policy_logits_jit, value_jit = inference_jit(model, params, test_input)
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

if __name__ == "__main__":
    test_swin_shogi_model() 