"""
将棋用のSwin Transformerモデル実装
"""
import os
import jax.numpy as jnp
import numpy as np
import flax
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
        if config.norm_layer is not None:
            self.norm = config.norm_layer()
        else:
            self.norm = flax.linen.LayerNorm()
        
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
    
    def __call__(self, x, feature_vector=None, deterministic: bool = True):
        """
        モデル推論を実行する

        Args:
            x: 盤面特徴量 [B, H, W, C]
            feature_vector: 手番・持ち駒特徴量 [B, feature_dim]
            deterministic: 決定論的実行フラグ（学習時はFalse）
            
        Returns:
            (policy_logits, value): 方策ロジットと価値
        """
        # feature_vectorがNoneの場合は0埋め特徴ベクトルを作成（テスト用）
        if feature_vector is None:
            B = x.shape[0]
            feature_vector = jnp.zeros((B, self.model_config.feature_dim))
        
        # 特徴抽出
        features = self._forward_features(x, feature_vector, deterministic)
        
        # ヘッド
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
    
    def _forward_features(self, x, feature_vector, deterministic: bool = True):
        """
        特徴抽出のための内部関数
        
        Args:
            x: 盤面特徴量 [B, H, W, C]
            feature_vector: 手番・持ち駒特徴量 [B, feature_dim]
            deterministic: 決定論的実行フラグ
            
        Returns:
            特徴ベクトル
        """
        # パッチ埋め込み
        x = self.patch_embed(x, feature_vector=feature_vector, deterministic=deterministic)
        
        B, L, C = x.shape
        
        # 先頭の特徴トークンを分離
        feature_token = x[:, :1]  # (B, 1, C)
        patch_tokens = x[:, 1:]   # (B, L-1, C)
        
        # パッチのみに位置埋め込みを適用
        patch_tokens = patch_tokens + self.absolute_pos_embed
        patch_tokens = self.pos_drop(patch_tokens, deterministic=deterministic)
        
        # 再結合
        x = jnp.concatenate([feature_token, patch_tokens], axis=1)
        
        # 現在の解像度を計算
        config = self.model_config
        current_resolution = (
            config.img_size[0] // config.patch_size[0],
            config.img_size[1] // config.patch_size[1]
        )
        
        # Swin Transformer段階
        for layer in self.layers.values():
            x, current_resolution = layer(x, current_resolution=current_resolution, 
                                        deterministic=deterministic)
        
        # 最終正規化
        x = self.norm(x)
        
        # 特徴トークンを使用
        x = x[:, 0]  # 先頭の特徴トークンのみ使用 (ViTのCLSトークンと同様)
        
        return x
    
    def extract_features(self, x, feature_vector=None, deterministic: bool = True):
        """
        中間特徴を抽出する（Actor-Criticで使用）
        シーケンス図のTransformer->AC部分の実装
        
        Args:
            x: 盤面特徴量 [B, H, W, C]
            feature_vector: 手番・持ち駒特徴量 [B, feature_dim]
            deterministic: 決定論的実行フラグ
            
        Returns:
            特徴ベクトル
        """
        return self._forward_features(x, feature_vector, deterministic)
    
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