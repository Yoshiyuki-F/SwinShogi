"""将棋の盤面表示用のユーティリティ"""

from typing import Dict, List, Tuple, Optional, Union
import io

from .shogi_pieces import Player, PieceType, ShogiPiece

class BoardVisualizer:
    """将棋の盤面表示クラス"""
    
    @staticmethod
    def get_piece_symbols() -> Dict[Tuple[PieceType, Player], str]:
        """
        駒の文字表現を取得
        
        Returns:
            Dict: 駒の種類と所有者に対応する文字表現
        """
        return {
            (PieceType.PAWN, Player.SENTE): '歩',
            (PieceType.LANCE, Player.SENTE): '香',
            (PieceType.KNIGHT, Player.SENTE): '桂',
            (PieceType.SILVER, Player.SENTE): '銀',
            (PieceType.GOLD, Player.SENTE): '金',
            (PieceType.BISHOP, Player.SENTE): '角',
            (PieceType.ROOK, Player.SENTE): '飛',
            (PieceType.KING, Player.SENTE): '玉',
            (PieceType.PROMOTED_PAWN, Player.SENTE): 'と',
            (PieceType.PROMOTED_LANCE, Player.SENTE): '杏',
            (PieceType.PROMOTED_KNIGHT, Player.SENTE): '圭',
            (PieceType.PROMOTED_SILVER, Player.SENTE): '全',
            (PieceType.PROMOTED_BISHOP, Player.SENTE): '馬',
            (PieceType.PROMOTED_ROOK, Player.SENTE): '龍',
            
            (PieceType.PAWN, Player.GOTE): '↓歩',
            (PieceType.LANCE, Player.GOTE): '↓香',
            (PieceType.KNIGHT, Player.GOTE): '↓桂',
            (PieceType.SILVER, Player.GOTE): '↓銀',
            (PieceType.GOLD, Player.GOTE): '↓金',
            (PieceType.BISHOP, Player.GOTE): '↓角',
            (PieceType.ROOK, Player.GOTE): '↓飛',
            (PieceType.KING, Player.GOTE): '↓玉',
            (PieceType.PROMOTED_PAWN, Player.GOTE): '↓と',
            (PieceType.PROMOTED_LANCE, Player.GOTE): '↓杏',
            (PieceType.PROMOTED_KNIGHT, Player.GOTE): '↓圭',
            (PieceType.PROMOTED_SILVER, Player.GOTE): '↓全',
            (PieceType.PROMOTED_BISHOP, Player.GOTE): '↓馬',
            (PieceType.PROMOTED_ROOK, Player.GOTE): '↓龍'
        }
    
    @staticmethod
    def get_piece_type_from_name(piece_name: str) -> Optional[PieceType]:
        """
        駒の名前からPieceTypeを取得
        
        Args:
            piece_name: 駒の名前（例: "pawn", "lance"）
            
        Returns:
            PieceType: 対応するPieceType、見つからない場合はNone
        """
        name_to_type = {
            "pawn": PieceType.PAWN,
            "lance": PieceType.LANCE,
            "knight": PieceType.KNIGHT,
            "silver": PieceType.SILVER,
            "gold": PieceType.GOLD,
            "bishop": PieceType.BISHOP,
            "rook": PieceType.ROOK,
            "king": PieceType.KING,
            "promoted_pawn": PieceType.PROMOTED_PAWN,
            "promoted_lance": PieceType.PROMOTED_LANCE,
            "promoted_knight": PieceType.PROMOTED_KNIGHT,
            "promoted_silver": PieceType.PROMOTED_SILVER,
            "promoted_bishop": PieceType.PROMOTED_BISHOP,
            "promoted_rook": PieceType.PROMOTED_ROOK
        }
        return name_to_type.get(piece_name)
    
    @classmethod
    def board_to_string(cls, 
                        board: List[List[Optional[ShogiPiece]]], 
                        hands: Dict[Player, Dict[str, int]], 
                        turn: Player) -> str:
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
        
        # 駒の文字表現
        piece_symbols = cls.get_piece_symbols()
        
        # 行番号を漢字で表現
        row_kanji = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
        
        # 列のラベル
        output.write('  ９   ８    ７    ６   ５   ４   ３   ２    １  \n')
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
                    symbol = piece_symbols.get((piece.piece_type, piece.player), '??')
                    
                    # 後手の駒（矢印がある駒）はスペースなしで表示
                    if piece.player == Player.GOTE:
                        row_str += f'{symbol} |'
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
                    piece_type = cls.get_piece_type_from_name(piece_name)
                    if piece_type:
                        symbol = piece_symbols.get((piece_type, Player.SENTE), '??')
                        sente_hand_str += f'{symbol}{count} '
        output.write(sente_hand_str + '\n')
        
        # 後手の持ち駒
        gote_hand_str = '後手: '
        if Player.GOTE in hands:
            for piece_name, count in hands[Player.GOTE].items():
                if count > 0:
                    piece_type = cls.get_piece_type_from_name(piece_name)
                    if piece_type:
                        symbol = piece_symbols.get((piece_type, Player.GOTE), '??')
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