"""
将棋のルールと駒の動きを実装するモジュール
"""
from typing import List, Tuple
from enum import Enum, auto
from dataclasses import dataclass

class Player(Enum):
    """プレイヤー（先手/後手）"""
    SENTE = 0  # 先手
    GOTE = 1   # 後手

class PieceType(Enum):
    """駒の種類"""
    # 基本駒
    PAWN = auto()      # 歩兵
    LANCE = auto()     # 香車
    KNIGHT = auto()    # 桂馬
    SILVER = auto()    # 銀将
    GOLD = auto()      # 金将
    BISHOP = auto()    # 角行
    ROOK = auto()      # 飛車
    KING = auto()      # 玉将
    
    # 成り駒
    PROMOTED_PAWN = auto()      # と金
    PROMOTED_LANCE = auto()     # 成香
    PROMOTED_KNIGHT = auto()    # 成桂
    PROMOTED_SILVER = auto()    # 成銀
    PROMOTED_BISHOP = auto()    # 馬
    PROMOTED_ROOK = auto()      # 龍

@dataclass
class ShogiPiece:
    """将棋の駒を表すクラス"""
    piece_type: PieceType
    player: Player
    promoted: bool = False
    
    def __post_init__(self):
        """初期化後の処理"""
        # 成り駒の場合はpromotedフラグを設定
        if self.piece_type in [
            PieceType.PROMOTED_PAWN, PieceType.PROMOTED_LANCE, 
            PieceType.PROMOTED_KNIGHT, PieceType.PROMOTED_SILVER,
            PieceType.PROMOTED_BISHOP, PieceType.PROMOTED_ROOK
        ]:
            self.promoted = True
    
    @property
    def name(self) -> str:
        """駒の名前を取得"""
        name_map = {
            PieceType.PAWN: "pawn",
            PieceType.LANCE: "lance", 
            PieceType.KNIGHT: "knight",
            PieceType.SILVER: "silver",
            PieceType.GOLD: "gold",
            PieceType.BISHOP: "bishop",
            PieceType.ROOK: "rook",
            PieceType.KING: "king",
            PieceType.PROMOTED_PAWN: "promoted_pawn",
            PieceType.PROMOTED_LANCE: "promoted_lance",
            PieceType.PROMOTED_KNIGHT: "promoted_knight", 
            PieceType.PROMOTED_SILVER: "promoted_silver",
            PieceType.PROMOTED_BISHOP: "promoted_bishop",
            PieceType.PROMOTED_ROOK: "promoted_rook"
        }
        return name_map[self.piece_type]
    
    @property
    def base_piece_type(self) -> PieceType:
        """元の駒の種類を取得（成り駒の場合）"""
        if not self.promoted:
            return self.piece_type
            
        base_map = {
            PieceType.PROMOTED_PAWN: PieceType.PAWN,
            PieceType.PROMOTED_LANCE: PieceType.LANCE,
            PieceType.PROMOTED_KNIGHT: PieceType.KNIGHT,
            PieceType.PROMOTED_SILVER: PieceType.SILVER,
            PieceType.PROMOTED_BISHOP: PieceType.BISHOP,
            PieceType.PROMOTED_ROOK: PieceType.ROOK
        }
        return base_map.get(self.piece_type, self.piece_type)
    
    def can_promote(self) -> bool:
        """駒が成れるかどうかを判定"""
        if self.promoted:
            return False
            
        # 金と玉は成れない
        if self.piece_type in [PieceType.GOLD, PieceType.KING]:
            return False
            
        return True
    
    def get_promoted_piece_type(self) -> PieceType:
        """成り駒の種類を取得"""
        if not self.can_promote():
            return self.piece_type
            
        promote_map = {
            PieceType.PAWN: PieceType.PROMOTED_PAWN,
            PieceType.LANCE: PieceType.PROMOTED_LANCE,
            PieceType.KNIGHT: PieceType.PROMOTED_KNIGHT,
            PieceType.SILVER: PieceType.PROMOTED_SILVER,
            PieceType.BISHOP: PieceType.PROMOTED_BISHOP,
            PieceType.ROOK: PieceType.PROMOTED_ROOK
        }
        return promote_map.get(self.piece_type, self.piece_type)
    
    def get_moves(self, position: Tuple[int, int], board) -> List[Tuple[int, int]]:
        """
        駒の移動可能な位置のリストを返す
        
        Args:
            position: 現在の位置 (row, col)
            board: 盤面
            
        Returns:
            移動可能な位置のリスト [(row, col), ...]
        """
        row, col = position
        moves = []
        
        # 駒の種類に応じた移動処理
        if self.piece_type == PieceType.PAWN:
            # 歩兵
            if self.player == Player.SENTE:  # 先手
                moves.append((row - 1, col))
            else:  # 後手
                moves.append((row + 1, col))
                
        elif self.piece_type == PieceType.LANCE:
            # 香車
            direction = -1 if self.player == Player.SENTE else 1
            for i in range(1, 9):
                new_row = row + i * direction
                if 0 <= new_row < 9:
                    moves.append((new_row, col))
                    if board[new_row][col] is not None:
                        break
                else:
                    break
                    
        elif self.piece_type == PieceType.KNIGHT:
            # 桂馬
            if self.player == Player.SENTE:  # 先手
                moves.extend([(row - 2, col - 1), (row - 2, col + 1)])
            else:  # 後手
                moves.extend([(row + 2, col - 1), (row + 2, col + 1)])
                
        elif self.piece_type == PieceType.SILVER:
            # 銀
            if self.player == Player.SENTE:  # 先手
                moves.extend([
                    (row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
                    (row + 1, col - 1), (row + 1, col + 1)
                ])
            else:  # 後手
                moves.extend([
                    (row + 1, col - 1), (row + 1, col), (row + 1, col + 1),
                    (row - 1, col - 1), (row - 1, col + 1)
                ])
                
        elif self.piece_type == PieceType.GOLD or self.promoted:
            # 金または成り駒
            if self.player == Player.SENTE:  # 先手
                moves.extend([
                    (row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
                    (row, col - 1), (row, col + 1),
                    (row + 1, col)
                ])
            else:  # 後手
                moves.extend([
                    (row + 1, col - 1), (row + 1, col), (row + 1, col + 1),
                    (row, col - 1), (row, col + 1),
                    (row - 1, col)
                ])
                
        elif self.piece_type == PieceType.BISHOP:
            # 角
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in directions:
                for i in range(1, 9):
                    new_row, new_col = row + i * dr, col + i * dc
                    if 0 <= new_row < 9 and 0 <= new_col < 9:
                        moves.append((new_row, new_col))
                        if board[new_row][col] is not None:
                            break
                    else:
                        break
                        
            # 成り角（馬）の場合は十字方向にも1マス移動可能
            if self.promoted:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 9 and 0 <= new_col < 9:
                        moves.append((new_row, new_col))
                
        elif self.piece_type == PieceType.ROOK:
            # 飛車
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dr, dc in directions:
                for i in range(1, 9):
                    new_row, new_col = row + i * dr, col + i * dc
                    if 0 <= new_row < 9 and 0 <= new_col < 9:
                        moves.append((new_row, new_col))
                        if board[new_row][col] is not None:
                            break
                    else:
                        break
                        
            # 成り飛車（龍）の場合は斜め方向にも1マス移動可能
            if self.promoted:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 9 and 0 <= new_col < 9:
                        moves.append((new_row, new_col))
                
        elif self.piece_type == PieceType.KING:
            # 玉/王
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 9 and 0 <= new_col < 9:
                        moves.append((new_row, new_col))
        
        # 盤外や自分の駒があるマスを除外
        valid_moves = []
        for r, c in moves:
            if 0 <= r < 9 and 0 <= c < 9:
                if board[r][c] is None or board[r][c].player != self.player:
                    valid_moves.append((r, c))
                    
        return valid_moves
    
    def can_promote_at_position(self, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """
        駒が成れるかどうかを判定
        
        Args:
            move: 移動元と移動先の座標 ((from_row, from_col), (to_row, to_col))
            
        Returns:
            成れるかどうか
        """
        if not self.can_promote():
            return False
            
        from_pos, to_pos = move
        _, _ = from_pos
        to_row, _ = to_pos
        
        # 移動先または移動元が敵陣なら成れる
        if self.player == Player.SENTE:  # 先手
            return to_row < 3
        else:  # 後手
            return to_row > 5
    
    def must_promote(self, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """
        駒が成らなければならないかどうかを判定
        
        Args:
            move: 移動元と移動先の座標 ((from_row, from_col), (to_row, to_col))
            
        Returns:
            成らなければならないかどうか
        """
        if self.promoted or self.piece_type not in [PieceType.PAWN, PieceType.LANCE, PieceType.KNIGHT]:
            return False
            
        _, to_pos = move
        to_row, _ = to_pos
        
        if self.player == Player.SENTE:  # 先手
            if self.piece_type in [PieceType.PAWN, PieceType.LANCE]:
                return to_row == 0
            elif self.piece_type == PieceType.KNIGHT:
                return to_row <= 1
        else:  # 後手
            if self.piece_type in [PieceType.PAWN, PieceType.LANCE]:
                return to_row == 8
            elif self.piece_type == PieceType.KNIGHT:
                return to_row >= 7
                
        return False
    
    def __str__(self) -> str:
        """文字列表現"""
        symbols = {
            PieceType.PAWN: "歩", PieceType.LANCE: "香", PieceType.KNIGHT: "桂", PieceType.SILVER: "銀",
            PieceType.GOLD: "金", PieceType.BISHOP: "角", PieceType.ROOK: "飛", PieceType.KING: "玉",
            PieceType.PROMOTED_PAWN: "と", PieceType.PROMOTED_LANCE: "成香", PieceType.PROMOTED_KNIGHT: "成桂",
            PieceType.PROMOTED_SILVER: "成銀", PieceType.PROMOTED_BISHOP: "馬", PieceType.PROMOTED_ROOK: "龍"
        }
        piece = symbols.get(self.piece_type, "?")
        if self.player == Player.GOTE:
            return f"v{piece}"
        return piece

