#!/bin/bash

# SwinShogiの全テストを実行するスクリプト

cd "$(dirname "$0")/.."
PYTHONPATH="$(pwd)"

echo "将棋ルールのテストを実行中..."
python -m src.tests.test_shogi_rules

echo "SwinShogiモデルのテストを実行中..."
python -m src.tests.test_swin_shogi

echo "パフォーマンス評価のテストを実行中..."
python -m src.tests.test_performance_evaluation 