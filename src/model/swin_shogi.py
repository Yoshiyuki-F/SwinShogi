"""
Swin Transformerを使用した将棋AIモデルの実装
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
from typing import Callable, Optional, Tuple
import time

class MLP(nn.Module):
    """多層パーセプトロン"""
    hidden_dim: int
    out_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(features=self.out_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x

class WindowAttention(nn.Module):
    """ウィンドウ内の注意機構"""
    dim: int
    window_size: Tuple[int, int]
    num_heads: int
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    def setup(self):
        self.scale = (self.dim // self.num_heads) ** -0.5
        self.qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(rate=self.attn_drop)
        self.proj = nn.Dense(self.dim)
        self.proj_dropout = nn.Dropout(rate=self.proj_drop)

    def __call__(self, x, mask=None, deterministic: bool = True):
        B, H, W, C = x.shape
        N = H * W
        x = x.reshape(B, N, C)

        # qkv投影
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意機構のスケーリング
        attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * self.scale

        # マスクの適用（あれば）
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B // nW, nW, self.num_heads, N, N)
            attn = attn + mask.reshape(1, nW, 1, N, N)
            attn = attn.reshape(B, self.num_heads, N, N)
        
        attn = nn.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn, deterministic=deterministic)

        # 出力投影
        x = jnp.transpose(attn @ v, (0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x, deterministic=deterministic)
        x = x.reshape(B, H, W, C)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformerブロック"""
    dim: int
    input_resolution: Tuple[int, int]
    num_heads: int
    window_size: int = 7
    shift_size: int = 0
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    norm_layer: Callable = nn.LayerNorm

    def setup(self):
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0

        self.norm1 = self.norm_layer()
        self.attn = WindowAttention(
            dim=self.dim,
            window_size=(self.window_size, self.window_size),
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
        )
        self.norm2 = self.norm_layer()
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(hidden_dim=mlp_hidden_dim, out_dim=self.dim, dropout_rate=self.drop)

        # シフト操作用のマスクを計算
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = jnp.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask = img_mask.at[:, h, w, :].set(cnt)
                    cnt += 1

            mask_windows = self._window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
            attn_mask = jnp.where(attn_mask != 0, -100.0, 0.0)
            self.attn_mask = attn_mask
        else:
            self.attn_mask = None

    def _window_partition(self, x, window_size):
        """入力をウィンドウに分割"""
        B, H, W, C = x.shape
        x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(-1, window_size, window_size, C)
        return windows

    def _window_reverse(self, windows, window_size, H, W):
        """ウィンドウを元のサイズに戻す"""
        B = windows.shape[0] // (H * W // window_size // window_size)
        x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(B, H, W, -1)
        return x

    def __call__(self, x, deterministic: bool = True):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, L: {L}, H: {H}, W: {W}"

        residual = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        # 循環シフト
        if self.shift_size > 0:
            shifted_x = jnp.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # ウィンドウ分割
        x_windows = self._window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask, deterministic=deterministic)

        # ウィンドウ結合
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, self.window_size, H, W)

        # 逆シフト
        if self.shift_size > 0:
            x = jnp.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        # FFN
        x = residual + x
        x = x + self.mlp(self.norm2(x), deterministic=deterministic)

        return x

class PatchEmbed(nn.Module):
    """画像をパッチに分割して埋め込み"""
    img_size: Tuple[int, int] = (224, 224)
    patch_size: Tuple[int, int] = (4, 4)
    in_chans: int = 3
    embed_dim: int = 96

    def setup(self):
        patches_resolution = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.proj = nn.Conv(
            features=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding=0
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        x = self.proj(x)
        return x

class BasicLayer(nn.Module):
    """基本レイヤー：複数のSwin Transformerブロックを含む"""
    dim: int
    input_resolution: Tuple[int, int]
    depth: int
    num_heads: int
    window_size: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    norm_layer: Callable = nn.LayerNorm
    downsample: Optional[Callable] = None
    use_checkpoint: bool = False

    def setup(self):
        # SwinTransformerブロックを構築
        self.blocks = [
            SwinTransformerBlock(
                dim=self.dim,
                input_resolution=self.input_resolution,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop,
                attn_drop=self.attn_drop,
                drop_path=self.drop_path,
                norm_layer=self.norm_layer
            )
            for i in range(self.depth)
        ]

        # パッチ結合レイヤー
        if self.downsample is not None:
            self.ds = self.downsample(
                input_resolution=self.input_resolution,
                dim=self.dim,
                norm_layer=self.norm_layer
            )
        else:
            self.ds = None

    def __call__(self, x, deterministic: bool = True):
        for blk in self.blocks:
            x = blk(x, deterministic=deterministic)
        
        if self.ds is not None:
            x = self.ds(x)
        return x

class PatchMerging(nn.Module):
    """パッチ結合（ダウンサンプリング）レイヤー"""
    input_resolution: Tuple[int, int]
    dim: int
    norm_layer: Callable = nn.LayerNorm

    def setup(self):
        self.reduction = nn.Dense(2 * self.dim)
        self.norm = self.norm_layer()

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, L: {L}, H: {H}, W: {W}"
        x = x.reshape(B, H, W, C)

        # パッチをグループ化
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = jnp.concatenate([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape(B, -1, 4 * C)  # B H/2*W/2 4*C

        # 線形射影
        x = self.norm(x)
        x = self.reduction(x)

        return x

class SwinShogiModel(nn.Module):
    """Swin Transformerベースの将棋モデル"""
    img_size: Tuple[int, int] = (9, 9)  # 将棋盤は9x9
    patch_size: Tuple[int, int] = (1, 1)  # 将棋は1マスずつパッチ化
    in_chans: int = 119  # 入力チャネル数（駒の種類、位置、持ち駒など）
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 3
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Callable = nn.LayerNorm
    patch_norm: bool = True
    n_policy_outputs: int = 9*9*27  # 政策出力（移動元×移動先×駒の種類）
    
    def setup(self):
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        
        # パッチ分割と埋め込み
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim
        )
        patches_resolution = self.patch_embed.patches_resolution
        
        # 絶対位置埋め込み
        self.absolute_pos_embed = self.param(
            'absolute_pos_embed',
            nn.initializers.normal(0.02),
            (1, patches_resolution[0] * patches_resolution[1], self.embed_dim)
        )
        
        # ドロップアウト
        self.pos_drop = nn.Dropout(rate=self.drop_rate)
        
        # 段階的なドロップパス率
        dpr = [x for x in np.linspace(0, self.drop_path_rate, sum(self.depths))]
        
        # ネットワークの段階的な構築
        # リストではなくModuleDictを使用
        self.layers = {
            f'layer{i}': BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            for i_layer, i in enumerate(range(self.num_layers))
        }
        
        # 最終正規化層
        self.norm = self.norm_layer()
        
        # 政策ヘッド（移動確率）
        self.policy_head = nn.Sequential([
            nn.Dense(512),
            nn.gelu,
            nn.Dense(self.n_policy_outputs)
        ])
        
        # 価値ヘッド（勝率評価）
        self.value_head = nn.Sequential([
            nn.Dense(256),
            nn.gelu,
            nn.Dense(1),
            nn.tanh
        ])
    
    def __call__(self, x, deterministic: bool = True):
        # パッチ埋め込み
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        
        # 位置埋め込みの追加
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x, deterministic=deterministic)
        
        # Swin Transformer段階
        for layer in self.layers.values():
            x = layer(x, deterministic=deterministic)
        
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
    
    def load_params(self, path: str):
        """モデルパラメータを読み込み"""
        with open(path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return params
    
    @staticmethod
    def cross_entropy_loss(logits, targets):
        """交差エントロピー損失関数"""
        return -jnp.sum(targets * jax.nn.log_softmax(logits)) / targets.shape[0]
    
    @jax.jit
    def policy_gradient_loss(params, model_apply_fn, inputs, targets):
        """方策勾配損失関数（JIT最適化）"""
        policy_logits, _ = model_apply_fn(params, inputs)
        return SwinShogiModel.cross_entropy_loss(policy_logits, targets)

def create_swin_shogi_model(rng, batch_size=1):
    """SwinShogiモデルの作成"""
    model = SwinShogiModel()
    input_shape = (batch_size, 9, 9, 119)  # B, H, W, C
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
    loaded_params = model.load_params(test_save_path)
    
    # 同一性確認
    equal = all(jnp.array_equal(p1, p2) for p1, p2 in zip(jax.tree_leaves(params), jax.tree_leaves(loaded_params)))
    print(f"パラメータの保存と読み込みテスト: {'成功' if equal else '失敗'}")
    
    return True

if __name__ == "__main__":
    test_swin_shogi_model()