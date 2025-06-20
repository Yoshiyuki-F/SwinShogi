#!/usr/bin/env python3
"""
SwinShogi - 将棋AIエンジン
"""
import argparse

def main():
    """SwinShogiのメインエントリーポイント"""
    parser = argparse.ArgumentParser(description='SwinShogi - 将棋AIエンジン')
    parser.add_argument('--usi', action='store_true', help='USIモードで起動')
    parser.add_argument('--train', action='store_true', help='学習モードで起動')
    parser.add_argument('--evaluate', action='store_true', help='評価モードで起動')
    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    
    args = parser.parse_args()
    
    if args.usi:
        print("USIモードで起動します")
        from .interface.usi import USIEngine
        engine = USIEngine()
        engine.start()
    elif args.train:
        print("学習モードで起動します")
        from .rl.shogi_rl import train
        train()
    elif args.evaluate:
        print("評価モードで起動します")
        from .evaluation.evaluator import evaluate
        evaluate()
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main() 