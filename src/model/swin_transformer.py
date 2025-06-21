"""
Swin Transformerの基本的な実装
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Tuple
from config.default_config import get_model_config

# configから値を取得
config = get_model_config()

class MLP(nn.Module):
    """多層パーセプトロン"""
    hidden_dim: int
    out_dim: int
    dropout_rate: float

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
    qkv_bias: bool
    attn_drop: float
    proj_drop: float

    def setup(self):
        self.scale = (self.dim // self.num_heads) ** -0.5
        self.qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(rate=self.attn_drop)
        self.proj = nn.Dense(self.dim)
        self.proj_dropout = nn.Dropout(rate=self.proj_drop)

    def __call__(self, x, mask=None, deterministic: bool = True):
        # 入力は既に (B', N, C) の形状であると想定
        # B'はバッチサイズ×ウィンドウ数、Nはウィンドウ内の要素数（window_size * window_size）
        B, N, C = x.shape

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
        
        # 出力は3次元のまま返す
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformerブロック"""
    dim: int
    input_resolution: Tuple[int, int]
    num_heads: int
    window_size: int
    mlp_ratio: float
    qkv_bias: bool
    drop: float
    attn_drop: float
    drop_path: float 
    norm_layer: Callable
    shift_size: int
    
    def setup(self):
        # shift_sizeを直接変更する代わりに、適切なshift_sizeを計算
        shift_size = self.shift_size

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
        if shift_size > 0:
            H, W = self.input_resolution
            img_mask = jnp.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask = img_mask.at[:, h, w, :].set(cnt)
                    cnt += 1

            # window_partitionメソッドは(windows, info)のタプルを返すので、最初の要素だけを使用
            mask_windows_tuple = self._window_partition(img_mask, self.window_size)
            mask_windows = mask_windows_tuple[0]  # タプルの最初の要素（ウィンドウ）を取得
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
            attn_mask = jnp.where(attn_mask != 0, -100.0, 0.0)
            self.attn_mask = attn_mask
        else:
            self.attn_mask = None
            
        # シフトサイズを保存（処理に使用）
        self.shift_size_value = shift_size

    def _window_partition(self, x, window_size):
        """入力をウィンドウに分割"""
        B, H, W, C = x.shape
        
        # 入力サイズがウィンドウサイズで割り切れるかチェック
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        
        # 必要に応じてパディング
        if pad_h > 0 or pad_w > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
        
        # パディング後のサイズ
        H_padded, W_padded = H + pad_h, W + pad_w
        
        # ウィンドウ分割
        x = x.reshape(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
        windows = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(-1, window_size, window_size, C)
        return windows, (H_padded, W_padded, pad_h, pad_w)

    def _window_reverse(self, windows, window_size, H_padded, W_padded, pad_h=0, pad_w=0):
        """ウィンドウを元のサイズに戻す"""
        B = windows.shape[0] // (H_padded * W_padded // window_size // window_size)
        x = windows.reshape(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(B, H_padded, W_padded, -1)
        
        # パディングがある場合は元のサイズに戻す
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H_padded-pad_h, :W_padded-pad_w, :]
        
        return x

    def __call__(self, x, current_resolution=None, deterministic: bool = True):
        # 現在の解像度が指定されていなければデフォルトを使用
        H, W = current_resolution if current_resolution is not None else self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, L: {L}, H: {H}, W: {W}"

        residual = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        # 循環シフト
        if self.shift_size_value > 0:
            shifted_x = jnp.roll(x, shift=(-self.shift_size_value, -self.shift_size_value), axis=(1, 2))
        else:
            shifted_x = x

        # ウィンドウ分割
        x_windows_tuple = self._window_partition(shifted_x, self.window_size)
        x_windows = x_windows_tuple[0]  # 最初の要素（ウィンドウ）を取得
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask, deterministic=deterministic)

        # ウィンドウ結合
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        # 解像度情報を取得
        padded_info = x_windows_tuple[1]
        H_padded, W_padded = padded_info[0], padded_info[1]
        pad_h, pad_w = padded_info[2], padded_info[3]
        shifted_x = self._window_reverse(attn_windows, self.window_size, H_padded, W_padded, pad_h, pad_w)

        # 逆シフト
        if self.shift_size_value > 0:
            x = jnp.roll(shifted_x, shift=(self.shift_size_value, self.shift_size_value), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        # FFN
        x = residual + x
        x = x + self.mlp(self.norm2(x), deterministic=deterministic)

        return x

class PatchEmbed(nn.Module):
    """画像をパッチに分割して埋め込み"""
    img_size: Tuple[int, int]
    patch_size: Tuple[int, int]
    in_chans: int
    embed_dim: int

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
        
        # TODO: 将棋の持ち駒と手番を表す特殊トークンを追加
        # 1. 持ち駒特殊トークン
        #   - 先手の持ち駒: 7種類 (歩、香、桂、銀、金、角、飛) * 各駒の最大数
        #   - 後手の持ち駒: 7種類 (歩、香、桂、銀、金、角、飛) * 各駒の最大数
        # 2. 手番特殊トークン
        #   - 1次元の値 (先手=1, 後手=0)
        #
        # 実装方針:
        # 1. board_encoder.pyのget_pieces_in_hand_vectorとget_player_turn_vector関数を使用
        # 2. 特殊トークンを線形変換してembed_dimに射影
        # 3. 特殊トークンを入力シーケンスの先頭に追加
        #
        # self.hand_pieces_embed = nn.Dense(features=self.embed_dim)  # 持ち駒ベクトルの次元をembed_dimに射影
        # self.turn_embed = nn.Dense(features=self.embed_dim)  # 手番ベクトルの次元をembed_dimに射影

    def __call__(self, x, hand_pieces_vector=None, turn_vector=None):
        B, H, W, C = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        x = self.proj(x)
        x = x.reshape(B, -1, self.embed_dim)  # (B, H*W, C)
        
        # TODO: 実装する - 持ち駒と手番の特殊トークンを追加
        # if hand_pieces_vector is not None and turn_vector is not None:
        #     # 持ち駒ベクトルをembed_dim次元に変換
        #     hand_token = self.hand_pieces_embed(hand_pieces_vector)  # (B, embed_dim)
        #     hand_token = hand_token[:, None, :]  # (B, 1, embed_dim)
        #     
        #     # 手番ベクトルをembed_dim次元に変換
        #     turn_token = self.turn_embed(turn_vector)  # (B, embed_dim)
        #     turn_token = turn_token[:, None, :]  # (B, 1, embed_dim)
        #     
        #     # 特殊トークンを入力シーケンスの先頭に追加
        #     x = jnp.concatenate([hand_token, turn_token, x], axis=1)  # (B, 2+H*W, embed_dim)
        
        return x

class BasicLayer(nn.Module):
    """基本レイヤー：複数のSwin Transformerブロックを含む"""
    dim: int
    input_resolution: Tuple[int, int]
    depth: int
    num_heads: int
    window_size: int
    mlp_ratio: float
    qkv_bias: bool
    drop: float
    attn_drop: float
    drop_path: float
    norm_layer: Callable
    downsample: Optional[Callable]
    use_checkpoint: bool

    def setup(self):
        # SwinTransformerブロックを構築
        self.blocks = [
            SwinTransformerBlock(
                dim=self.dim,
                input_resolution=self.input_resolution,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=0 if (i % config.patch_merge_factor == 0) else self.window_size // config.patch_merge_factor,
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

    def __call__(self, x, current_resolution=None, deterministic: bool = True):
        # 現在の解像度が指定されていなければデフォルトを使用
        resolution = current_resolution if current_resolution is not None else self.input_resolution
        
        # 各ブロックに現在の解像度を渡す
        for _, blk in enumerate(self.blocks):
            x = blk(x, current_resolution=resolution, deterministic=deterministic)
        
        if self.ds is not None:
            x, new_resolution = self.ds(x)
            return x, new_resolution
        return x, resolution

class PatchMerging(nn.Module):
    """パッチ結合（ダウンサンプリング）レイヤー"""
    input_resolution: Tuple[int, int]
    dim: int
    norm_layer: Callable

    def setup(self):
        # パッチ結合係数を設定から取得
        self.patch_merge_factor = config.patch_merge_factor
        # パッチ結合係数の2乗倍の次元になる
        merge_factor_squared = self.patch_merge_factor * self.patch_merge_factor
        self.reduction = nn.Dense(merge_factor_squared * self.dim)
        self.norm = self.norm_layer()

    def __call__(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, L: {L}, H: {H}, W: {W}"
        x = x.reshape(B, H, W, C)

        # パッチ結合係数で割り切れるかチェック
        factor = self.patch_merge_factor
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        if pad_h > 0 or pad_w > 0:
            # 必要に応じて下と右にパディング
            x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W

        # factor×factorのグループに分割
        x_groups = []
        for i in range(factor):
            for j in range(factor):
                x_groups.append(x[:, i::factor, j::factor, :])  # B H/factor W/factor C
        
        # すべてのグループを連結
        x = jnp.concatenate(x_groups, -1)  # B H/factor W/factor factor*factor*C
        
        # 新しい解像度を計算
        H_new, W_new = H_padded // factor, W_padded // factor
        x = x.reshape(B, H_new * W_new, factor * factor * C)  # B H_new*W_new factor*factor*C

        # 線形射影
        x = self.norm(x)
        x = self.reduction(x)

        return x, (H_new, W_new) 