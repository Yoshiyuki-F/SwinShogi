"""
SwinShogiテストパッケージ

このパッケージには、SwinShogiの各モジュールのテストが含まれています。
テストは以下のカテゴリに分類されています：

- model: モデル関連のテスト
- shogi: 将棋ルール関連のテスト
- rl: 強化学習関連のテスト
- utils: ユーティリティ関連のテスト
- interface: インターフェース関連のテスト
"""

from .test_swin_shogi import TestSwinShogi
from .test_shogi_rules import TestShogiRules
from .test_performance_evaluation import TestPerformanceEvaluation, TestJAXOptimizations

__all__ = [
    'TestSwinShogi',
    'TestShogiRules',
    'TestPerformanceEvaluation',
    'TestJAXOptimizations'
]
