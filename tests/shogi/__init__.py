"""
将棋モジュールのテスト

このパッケージには、SwinShogiの将棋ルール関連のテストが含まれています。
盤面表現や合法手生成、ルール適用のテストが含まれます。
"""

from .test_shogi_rules import TestShogiRules

__all__ = ['TestShogiRules']
