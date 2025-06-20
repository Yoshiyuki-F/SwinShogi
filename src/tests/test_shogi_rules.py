import unittest
from src.shogi.shogi_pieces import ShogiGame

class TestShogiRules(unittest.TestCase):
    def setUp(self):
        self.game = ShogiGame()
        
    def test_initial_board(self):
        """初期盤面のテスト"""
        board = self.game.board
        self.assertEqual(len(board), 9)
        self.assertEqual(len(board[0]), 9)
        
        # 駒の初期配置を確認
        self.assertEqual(board[0][0].name, "lance")
        self.assertEqual(board[0][1].name, "knight")
        self.assertEqual(board[0][2].name, "silver")
        self.assertEqual(board[0][3].name, "gold")
        self.assertEqual(board[0][4].name, "king")
        
    def test_valid_moves(self):
        """有効な手のテスト"""
        valid_moves = self.game.get_valid_moves()
        # 初期状態では歩兵の移動などの合法手が存在する
        self.assertGreater(len(valid_moves), 0)
        
    def test_move_piece(self):
        """駒の移動テスト"""
        # 歩兵を前進させる
        initial_state = self.game.get_state_hash()
        move = "7g7f"  # 例：７七の歩を７六に移動
        self.game.move(move)
        
        # 移動後は状態が変わっていることを確認
        self.assertNotEqual(initial_state, self.game.get_state_hash())
        
    def test_promotion(self):
        """駒の成りテスト"""
        # テスト用に特定の盤面を設定
        self.game.setup_custom_position("8/9/9/9/9/9/9/1P7/9 b - 1")
        
        # １二に歩兵を移動して成る
        move = "1b1a+"
        self.game.move(move)
        
        # １一の位置に成金があることを確認
        self.assertEqual(self.game.board[0][0].name, "promoted_pawn")
        
    def test_check_detection(self):
        """王手の判定テスト"""
        # 王手状態の盤面を設定
        self.game.setup_custom_position("k8/1G7/9/9/9/9/9/9/9 b - 1")
        
        # 王手状態であることを確認
        self.assertTrue(self.game.is_check())
        
    def test_checkmate_detection(self):
        """詰みの判定テスト"""
        # 詰み状態の盤面を設定
        self.game.setup_custom_position("k8/1G1G6/9/9/9/9/9/9/9 b - 1")
        
        # 詰み状態であることを確認
        self.assertTrue(self.game.is_checkmate())
        
    def test_pawn_drop_mate_rule(self):
        """打ち歩詰め禁止ルールのテスト"""
        # 打ち歩詰めができる盤面を設定
        self.game.setup_custom_position("k8/9/9/9/9/9/9/9/9 b P 1")
        
        # 打ち歩詰めの手を取得
        pawn_drop_mate_move = "1a1b"
        
        # 打ち歩詰めは禁止されているため、この手は有効な手に含まれていないはず
        valid_moves = self.game.get_valid_moves()
        self.assertNotIn(pawn_drop_mate_move, valid_moves)
        
    def test_repetition_draw(self):
        """千日手の判定テスト"""
        # テスト用に単純な盤面を設定
        self.game.setup_custom_position("k8/s8/9/9/9/9/9/8S/8K b - 1")
        
        # 同じ動きを4回繰り返して千日手を発生させる
        for _ in range(4):
            self.game.move("9i9h")  # 銀を上に移動
            self.game.move("9h9i")  # 銀を下に戻す
            self.game.move("1a1b")  # 相手の銀を上に移動
            self.game.move("1b1a")  # 相手の銀を下に戻す
            
        # 千日手が検出されるか確認
        self.assertTrue(self.game.is_repetition_draw())

if __name__ == "__main__":
    unittest.main() 