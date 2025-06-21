#!/bin/bash

# SwinShogi統合テストスクリプト
# SwinTransformer、Actor-Critic、MCTSの連携をテストします

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Pythonのパスにプロジェクトルートを追加
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 色付きの出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

echo -e "${BLUE}SwinShogi統合テストを実行します${RESET}"
echo -e "${YELLOW}Transformer、Actor-Critic、MCTSの連携をテストします${RESET}"
echo "==============================================="

# 利用可能なGPUをチェック
if [ -x "$(command -v nvidia-smi)" ]; then
  echo -e "${GREEN}GPUが利用可能です:${RESET}"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
  echo "-----------------------------------------------"
else
  echo -e "${YELLOW}GPU情報を取得できません。CPUで実行します。${RESET}"
  echo "-----------------------------------------------"
fi

# JAXログレベルを設定（警告のみ表示）
export JAX_LOG_LEVEL=WARNING

# MCTSのテスト実行
echo -e "${BLUE}MCTS単体テスト:${RESET}"
cd "$PROJECT_ROOT" && python -m unittest tests.rl.test_mcts

# モデルテスト実行
echo -e "\n${BLUE}SwinShogiモデルテスト:${RESET}"
cd "$PROJECT_ROOT" && python -m unittest src.tests.model.test_swin_shogi.TestSwinShogi

# 統合テスト実行
echo -e "\n${BLUE}SwinShogi統合テスト:${RESET}"
cd "$PROJECT_ROOT" && python -m unittest src.tests.model.test_swin_shogi.TestSwinShogiMCTSIntegration

# 実行結果に応じてメッセージ表示
if [ $? -eq 0 ]; then
  echo -e "\n${GREEN}テスト成功: SwinTransformer、Actor-Critic、MCTSが正常に連携しています${RESET}"
else
  echo -e "\n${RED}テスト失敗: 統合に問題があります${RESET}"
  exit 1
fi

# 完了メッセージ
echo -e "\n${BLUE}SwinShogiのシーケンス図に基づく実装が正常に動作しています${RESET}"
echo "===============================================" 