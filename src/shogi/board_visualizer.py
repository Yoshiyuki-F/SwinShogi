"""将棋の盤面表示用のユーティリティ"""

from typing import Dict, List, Optional
import io
from .shogi_pieces import Player, ShogiPiece

class BoardVisualizer:
    """将棋の盤面表示クラス"""
    
    @staticmethod
    def board_to_string(
        board: List[List[Optional[ShogiPiece]]], 
        hands: Dict[Player, Dict[str, int]], 
        turn: Player
    ) -> str:
        """
        将棋の盤面を文字列として取得
        
        Args:
            board: 9x9の盤面（駒の配置）
            hands: 持ち駒の辞書
            turn: 現在の手番
            
        Returns:
            str: 盤面の文字列表現
        """
        output = io.StringIO()
        
        # 行番号を漢字で表現
        row_kanji = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
        
        # 列のラベル
        output.write('  ９   ８   ７   ６   ５   ４   ３   ２   １  \n')
        output.write('+----+----+----+----+----+----+----+----+----+\n')
        
        # 盤面表示
        for row in range(9):
            row_str = '|'
            for col in range(9):
                piece = board[row][col]
                if piece is None:
                    symbol = '・'
                    row_str += f' {symbol} |'
                else:
                    symbol = str(piece).replace('v', '')
                    
                    # 後手の駒はスペースなしで表示
                    if piece.player == Player.GOTE:
                        row_str += f'↓{symbol} |'
                    else:
                        # 先手はスペースを入れて中央揃え
                        row_str += f' {symbol} |'
                    
            output.write(row_str + f' {row_kanji[row]}\n')
            output.write('+----+----+----+----+----+----+----+----+----+\n')
        
        # 手番表示
        output.write(f"手番: {'先手' if turn == Player.SENTE else '後手'}\n")
        
        # 持ち駒表示
        output.write('持ち駒:\n')
        
        # 先手の持ち駒
        sente_hand_str = '先手: '
        gote_hand_str = '後手: '

        if Player.SENTE in hands:
            for piece_name, count in hands[Player.SENTE].items():
                if count > 0:
                    piece = ShogiPiece(piece_name, Player.SENTE)
                    symbol = piece.get_symbol(piece_name)
                    sente_hand_str += f'{symbol}{count} '
        
        if Player.GOTE in hands:
            for piece_name, count in hands[Player.GOTE].items():
                if count > 0:
                    piece = ShogiPiece(piece_name, Player.GOTE)
                    symbol = piece.get_symbol(piece_name)
                    gote_hand_str += f'{symbol}{count} '
        
        output.write(sente_hand_str + '\n')
        output.write(gote_hand_str + '\n')
        
        return output.getvalue()
    
    @classmethod
    def visualize_board(cls, state: Dict, model=None, params=None) -> None:
        """
        将棋の盤面をコンソールに表示する
        
        Args:
            state: 将棋環境の状態 (盤面、持ち駒、手番を含む)
            model: SwinTransformerモデル（評価表示用、オプション）
            params: モデルパラメータ（評価表示用、オプション）
        """
        board = state['board']
        hands = state['hands']
        turn = state['turn']
        
        # 盤面表示
        print(cls.board_to_string(board, hands, turn))
        
        # 盤面評価表示（モデルは常に提供されていなければならない）
        cls._display_position_evaluation(state)

    
    @classmethod
    def _display_position_evaluation(cls, state: Dict) -> None:
        """
        SwinTransformerによる盤面評価を表示する
        
        Args:
            state: 将棋環境の状態
        """
        # state に evaluation は常にある
        current_eval = state['evaluation']

        
        print("─" * 50)
        print("🧠 SwinTransformer 局面評価:")
        print(f"  手番側評価: {current_eval:+6.2f}")
        
        # 評価の解釈を表示
        if abs(current_eval) >= 1000:
            if current_eval > 0:
                print(f"  → 手番側勝勢（詰みあり）")
            else:
                print(f"  → 手番側劣勢（詰まされている）")
        elif current_eval > 10:
            print(f"  → 手番側有利")
        elif current_eval > 3:
            print(f"  → 手番側やや有利") 
        elif current_eval > -3:
            print(f"  → 互角")
        elif current_eval > -10:
            print(f"  → 手番側やや不利")
        else:
            print(f"  → 手番側不利")
        print("─" * 50) 