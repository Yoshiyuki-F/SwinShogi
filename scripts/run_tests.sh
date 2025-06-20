#!/bin/bash

# SwinShogiのテストを実行するスクリプト

cd "$(dirname "$0")/.."
PYTHONPATH="$(pwd)" python -m src.tests.test_swin_shogi 