"""将棋の盤面表示用のユーティリティ"""

from typing import Dict, List, Tuple, Optional, Union
import io
import os
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
        output.write(f"手番: {'先手(下)' if turn == Player.SENTE else '後手(上)'}\n")
        
        # 持ち駒表示
        output.write('持ち駒:\n')
        
        # 先手の持ち駒
        sente_hand_str = '先手: '
        if Player.SENTE in hands:
            for piece_name, count in hands[Player.SENTE].items():
                if count > 0:
                    piece = ShogiPiece(piece_name, Player.SENTE)
                    symbol = piece.symbol
                    sente_hand_str += f'{symbol}{count} '
        output.write(sente_hand_str + '\n')
        
        # 後手の持ち駒
        gote_hand_str = '後手: '
        if Player.GOTE in hands:
            for piece_name, count in hands[Player.GOTE].items():
                if count > 0:
                    piece = ShogiPiece(piece_name, Player.GOTE)
                    symbol = piece.symbol
                    gote_hand_str += f'{symbol}{count} '
        output.write(gote_hand_str + '\n')
        
        return output.getvalue()
    
    @classmethod
    def visualize_board(cls, state: Dict) -> None:
        """
        将棋の盤面をコンソールに表示する
        
        Args:
            state: 将棋環境の状態 (盤面、持ち駒、手番を含む)
        """
        board = state['board']
        hands = state['hands']
        turn = state['turn']
        
        print(cls.board_to_string(board, hands, turn))
    
    @classmethod
    def visualize_game(cls, game) -> None:
        """
        ShogiGameオブジェクトの盤面をコンソールに表示する
        
        Args:
            game: ShogiGameオブジェクト
        """
        state = {
            'board': game.board,
            'hands': game.captures,
            'turn': game.current_player
        }
        cls.visualize_board(state) 