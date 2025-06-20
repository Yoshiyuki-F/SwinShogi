"""
将棋のルールと駒の動きを実装するモジュール
"""
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum

class Player(Enum):
    """プレイヤー（先手/後手）"""
    SENTE = 0  # 先手
    GOTE = 1   # 後手

class ShogiPiece:
    """将棋の駒クラス"""
    Gold_directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)]
    Bishop_directions = [(-1, -1, "line"), (1, -1, "line"), (-1, 1, "line"), (1, 1, "line")]
    Rook_directions = [(0, -1, "line"), (-1, 0, "line"), (1, 0, "line"), (0, 1, "line")]
    
    # 駒の文字と情報の対応
    PIECE_CHAR_INFO = {
        'p': {
            "type": "pawn", 
            "symbol": "歩", 
            "can_promote": True, 
            "promoted_type": "promoted_pawn", 
            "directions": [(0, -1)]  # 前に1マス
        },
        'l': {
            "type": "lance", 
            "symbol": "香", 
            "can_promote": True, 
            "promoted_type": "promoted_lance", 
            "directions": [(0, -1, "line")]  # 前に何マスでも
        },
        'n': {
            "type": "knight", 
            "symbol": "桂", 
            "can_promote": True, 
            "promoted_type": "promoted_knight", 
            "directions": [(-1, -2), (1, -2)]  # 前に2マス、横に1マス
        },
        's': {
            "type": "silver", 
            "symbol": "銀", 
            "can_promote": True, 
            "promoted_type": "promoted_silver", 
            "directions": [(-1, -1), (0, -1), (1, -1), (-1, 1), (1, 1)]  # 銀の動き
        },
        'g': {
            "type": "gold", 
            "symbol": "金", 
            "can_promote": False, 
            "promoted_type": None, 
            "directions": Gold_directions  # 金の動き
        },
        'b': {
            "type": "bishop", 
            "symbol": "角", 
            "can_promote": True, 
            "promoted_type": "promoted_bishop", 
            "directions": Bishop_directions  # 角の動き
        },
        'r': {
            "type": "rook", 
            "symbol": "飛", 
            "can_promote": True, 
            "promoted_type": "promoted_rook", 
            "directions": Rook_directions  # 飛車の動き
        },
        'k': {
            "type": "king", 
            "symbol": "玉", 
            "can_promote": False, 
            "promoted_type": None, 
            "directions": [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]  # 玉の動き
        },
        '+p': {
            "type": "promoted_pawn", 
            "symbol": "と", 
            "can_promote": False, 
            "base_type": "pawn", 
            "directions": Gold_directions  # 金と同じ動き
        },
        '+l': {
            "type": "promoted_lance", 
            "symbol": "成香", 
            "can_promote": False, 
            "base_type": "lance", 
            "directions": Gold_directions  # 金と同じ動き
        },
        '+n': {
            "type": "promoted_knight", 
            "symbol": "成桂", 
            "can_promote": False, 
            "base_type": "knight", 
            "directions": Gold_directions  # 金と同じ動き
        },
        '+s': {
            "type": "promoted_silver", 
            "symbol": "成銀", 
            "can_promote": False, 
            "base_type": "silver", 
            "directions": Gold_directions  # 金と同じ動き
        },
        '+b': {
            "type": "promoted_bishop", 
            "symbol": "馬", 
            "can_promote": False, 
            "base_type": "bishop", 
            "directions": Bishop_directions + [(0, -1), (-1, 0), (1, 0), (0, 1)]  # 追加の動き（十字1マス）
        },
        '+r': {
            "type": "promoted_rook", 
            "symbol": "龍", 
            "can_promote": False, 
            "base_type": "rook", 
            "directions": Rook_directions + [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # 追加の動き（斜め1マス）
        }
    }
    
    def __init__(self, piece_type: str, player: Player, promoted: bool = False):
        """
        駒の初期化
        
        Args:
            piece_type: 駒の種類（"pawn", "lance"など）
            player: プレイヤー（先手/後手）
            promoted: 成り駒かどうか
        """
        # 成り駒の場合はpiece_typeを調整
        if promoted and piece_type in self.get_promotable_types():
            self.piece_type = self.get_promoted_type(piece_type)
            self.promoted = True
        elif piece_type in self.get_all_types():
            self.piece_type = piece_type
            self.promoted = piece_type.startswith("promoted_")
        else:
            raise ValueError(f"不明な駒タイプ: {piece_type}")
            
        self.player = player
    
    @classmethod
    def get_all_types(cls):
        """すべての駒タイプのリストを取得"""
        return [info["type"] for info in cls.PIECE_CHAR_INFO.values()]
    
    @classmethod
    def get_promotable_types(cls):
        """成ることができる駒タイプのリストを取得"""
        return [info["type"] for info in cls.PIECE_CHAR_INFO.values() if info.get("can_promote", False)]
    
    @classmethod
    def get_promoted_type(cls, piece_type):
        """指定された駒タイプの成り駒タイプを取得"""
        for info in cls.PIECE_CHAR_INFO.values():
            if info["type"] == piece_type and info.get("can_promote", False):
                return info["promoted_type"]
        return None
    
    @classmethod
    def get_base_type(cls, piece_type):
        """指定された成り駒タイプの元の駒タイプを取得"""
        for info in cls.PIECE_CHAR_INFO.values():
            if info["type"] == piece_type and "base_type" in info:
                return info["base_type"]
        return piece_type
    
    @property
    def name(self) -> str:
        """駒の名前を取得"""
        return self.piece_type
    
    @property
    def __str__(self) -> str:
        """文字列表現"""
        symbol = self.get_symbol(self.piece_type)
        if self.player == Player.GOTE:
            return f"v{symbol}"
        return symbol
    
    @classmethod
    def get_symbol(cls, piece_type):
        """指定された駒タイプのシンボルを取得"""
        for info in cls.PIECE_CHAR_INFO.values():
            if info["type"] == piece_type:
                return info["symbol"]
        return "?"
    
    @property
    def can_promote(self) -> bool:
        """駒が成れるかどうかを判定"""
        for info in self.PIECE_CHAR_INFO.values():
            if info["type"] == self.piece_type:
                return info.get("can_promote", False)
        return False
    
    def get_promoted_piece(self) -> 'ShogiPiece':
        """成り駒を取得"""
        if not self.can_promote:
            return self
        
        promoted_type = None
        for info in self.PIECE_CHAR_INFO.values():
            if info["type"] == self.piece_type:
                promoted_type = info.get("promoted_type")
                break
                
        if promoted_type:
            return ShogiPiece(self.piece_type, self.player, promoted=True)
        
        return self
    
    @property
    def base_piece_type(self) -> str:
        """元の駒の種類を取得（成り駒の場合）"""
        return self.get_base_type(self.piece_type)
    
    def get_directions(self) -> List[Tuple]:
        """駒の移動方向を取得"""
        for info in self.PIECE_CHAR_INFO.values():
            if info["type"] == self.piece_type:
                return info["directions"]
        return []
    
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
        
        # 駒の移動方向を取得
        directions = self.get_directions()
        
        # 後手の場合は方向を反転
        if self.player == Player.GOTE:
            new_directions = []
            for d in directions:
                if len(d) == 2:
                    dx, dy = d
                    new_directions.append((-dx, -dy))
                else:
                    dx, dy, line = d
                    new_directions.append((-dx, -dy, line))
            directions = new_directions
        
        # 各方向について移動可能な位置を計算
        for direction in directions:
            if len(direction) == 3 and direction[2] == "line":
                # 直線移動（飛車、角、香車など）
                dx, dy = direction[0], direction[1]
                for i in range(1, 9):  # 最大8マス移動可能
                    new_row, new_col = row + dy * i, col + dx * i
                    if 0 <= new_row < 9 and 0 <= new_col < 9:
                        moves.append((new_row, new_col))
                        # 駒があれば、そこで止まる
                        if board[new_row][new_col] is not None:
                            break
                    else:
                        break
            else:
                # 1マス移動（金、銀、桂馬など）
                dx, dy = direction[0], direction[1]
                new_row, new_col = row + dy, col + dx
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
        if not self.can_promote:
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
        if self.promoted:
            return False
            
        _, to_pos = move
        to_row, _ = to_pos
        
        # 歩、香車、桂馬は特定の位置で成らなければならない
        if self.piece_type == "pawn" or self.piece_type == "lance":
            if self.player == Player.SENTE:  # 先手
                return to_row == 0
            else:  # 後手
                return to_row == 8
        elif self.piece_type == "knight":
            if self.player == Player.SENTE:  # 先手
                return to_row <= 1
            else:  # 後手
                return to_row >= 7
                
        return False

