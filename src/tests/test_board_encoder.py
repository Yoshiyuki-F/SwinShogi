"""
board_encoder.pyのテスト
"""

import unittest
import sys
import os

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.shogi.board_encoder import create_initial_board, encode_move, decode_move, encode_board_state
from src.shogi.board_visualizer import BoardVisualizer

class TestBoardEncoder(unittest.TestCase):
    """board_encoder.pyのテストクラス"""
    
    def test_create_initial_board(self):
        """初期盤面生成のテスト"""
        state = create_initial_board()
        
        # 状態に必要なキーが含まれているか確認
        self.assertIn('board', state)
        self.assertIn('hands', state)
        self.assertIn('turn', state)
        
        # 盤面の表示（デバッグ用）
        # BoardVisualizerは直接boardを受け取り、stateディクショナリは受け取らない
        board_str = BoardVisualizer.board_to_string(state['board'], state['hands'], state['turn'])
        print("初期盤面:")
        print(board_str)
        
        # 盤面のサイズが9x9であることを確認
        self.assertEqual(len(state['board']), 9)
        for row in state['board']:
            self.assertEqual(len(row), 9)
        
        # エンコードのテスト
        encoded = encode_board_state(state)
        self.assertIsNotNone(encoded)
        self.assertEqual(encoded.shape[1:], (9, 9))  # 高さと幅が9x9
    
    def test_move_encoding_decoding(self):
        """指し手のエンコード・デコードのテスト"""
        # 通常の手
        move = "7g7f"
        from_file = int(move[0])
        from_rank = ord(move[1]) - ord('a')
        to_file = int(move[2])
        to_rank = ord(move[3]) - ord('a')
        print(f"移動元: file={from_file} rank={from_rank} ({chr(ord('a') + from_rank)}), 移動先: file={to_file} rank={to_rank} ({chr(ord('a') + to_rank)})")
        
        encoded = encode_move(move)
        
        # エンコード結果の詳細を表示
        from_idx = encoded // 9
        to_idx = encoded % 9
        from_rank_calc = from_idx // 9
        from_col_calc = from_idx % 9
        to_rank_calc = to_idx // 9
        to_col_calc = to_idx % 9
        print(f"エンコード内部: from_idx={from_idx}, to_idx={to_idx}")
        print(f"計算結果: from_rank={from_rank_calc}, from_col={from_col_calc}, to_rank={to_rank_calc}, to_col={to_col_calc}")
        
        decoded = decode_move(encoded)
        print(f"通常の手: {move} -> encoded: {encoded} -> decoded: {decoded}")
        self.assertEqual(move, decoded)
        
        # 成る手
        move = "7g7f+"
        encoded = encode_move(move)
        decoded = decode_move(encoded)
        print(f"成る手: {move} -> encoded: {encoded} -> decoded: {decoded}")
        self.assertEqual(move, decoded)
        
        # 持ち駒を打つ手
        move = "P*5e"
        encoded = encode_move(move)
        decoded = decode_move(encoded)
        print(f"持ち駒を打つ手: {move} -> encoded: {encoded} -> decoded: {decoded}")
        self.assertEqual(move, decoded)

def main():
    """テストの実行"""
    unittest.main()

if __name__ == "__main__":
    main() 