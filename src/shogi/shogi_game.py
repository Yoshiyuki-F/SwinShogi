"""
将棋のゲームロジックを実装するモジュール
"""
from src.shogi.shogi_pieces import (
    ShogiPiece, Player
)
from typing import List, Tuple, Dict, Any, Optional
import re
import hashlib
from src.shogi.board_visualizer import BoardVisualizer
    
class ShogiGame:
    """将棋のゲームを表すクラス"""
    
    def __init__(self):
        """ゲームの初期化"""
        self.board = [[None for _ in range(9)] for _ in range(9)]
        self.captures = {Player.SENTE: {}, Player.GOTE: {}}  # 持ち駒
        self.current_player = Player.SENTE  # 先手から始める
        self.move_history = []  # 手の履歴
        self.position_history = []  # 局面の履歴
        self.setup_initial_board()
        
    def setup_initial_board(self):
        """初期配置の設定"""
        # 先手の駒配置（下側）
        self.board[8][0] = ShogiPiece("lance", Player.SENTE)
        self.board[8][1] = ShogiPiece("knight", Player.SENTE)
        self.board[8][2] = ShogiPiece("silver", Player.SENTE)
        self.board[8][3] = ShogiPiece("gold", Player.SENTE)
        self.board[8][4] = ShogiPiece("king", Player.SENTE)
        self.board[8][5] = ShogiPiece("gold", Player.SENTE)
        self.board[8][6] = ShogiPiece("silver", Player.SENTE)
        self.board[8][7] = ShogiPiece("knight", Player.SENTE)
        self.board[8][8] = ShogiPiece("lance", Player.SENTE)
        self.board[7][1] = ShogiPiece("bishop", Player.SENTE)
        self.board[7][7] = ShogiPiece("rook", Player.SENTE)
        for col in range(9):
            self.board[6][col] = ShogiPiece("pawn", Player.SENTE)
            
        # 後手の駒配置（上側）
        self.board[0][0] = ShogiPiece("lance", Player.GOTE)
        self.board[0][1] = ShogiPiece("knight", Player.GOTE)
        self.board[0][2] = ShogiPiece("silver", Player.GOTE)
        self.board[0][3] = ShogiPiece("gold", Player.GOTE)
        self.board[0][4] = ShogiPiece("king", Player.GOTE)
        self.board[0][5] = ShogiPiece("gold", Player.GOTE)
        self.board[0][6] = ShogiPiece("silver", Player.GOTE)
        self.board[0][7] = ShogiPiece("knight", Player.GOTE)
        self.board[0][8] = ShogiPiece("lance", Player.GOTE)
        self.board[1][1] = ShogiPiece("rook", Player.GOTE)
        self.board[1][7] = ShogiPiece("bishop", Player.GOTE)
        for col in range(9):
            self.board[2][col] = ShogiPiece("pawn", Player.GOTE)
            
        # 局面の履歴に現在の局面を追加
        self.position_history.append(self.get_state_hash())
        
    def setup_custom_position(self, sfen: str):
        """
        SFEN形式で指定された局面を設定
        
        Args:
            sfen: SFEN形式の盤面
        """
        # 盤面をクリア
        self.board = [[None for _ in range(9)] for _ in range(9)]
        self.captures = {Player.SENTE: {}, Player.GOTE: {}}
        
        parts = sfen.split()
        board_str = parts[0]
        turn = parts[1]
        captures = parts[2]
        
        # 盤面の設定
        rows = board_str.split('/')
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for char_idx, char in enumerate(row):
                if char.isdigit():
                    col_idx += int(char)
                else:
                    player = Player.GOTE if char.islower() else Player.SENTE
                    promoted = False
                    piece_char = char.lower()
                    
                    if piece_char == 'p':
                        piece_type = "pawn"
                    elif piece_char == 'l':
                        piece_type = "lance"
                    elif piece_char == 'n':
                        piece_type = "knight"
                    elif piece_char == 's':
                        piece_type = "silver"
                    elif piece_char == 'g':
                        piece_type = "gold"
                    elif piece_char == 'b':
                        piece_type = "bishop"
                    elif piece_char == 'r':
                        piece_type = "rook"
                    elif piece_char == 'k':
                        piece_type = "king"
                    elif piece_char == '+':
                        # 成り駒の処理
                        promoted = True
                        next_char = row[char_idx + 1].lower()
                        if next_char == 'p':
                            piece_type = "pawn"
                        elif next_char == 'l':
                            piece_type = "lance"
                        elif next_char == 'n':
                            piece_type = "knight"
                        elif next_char == 's':
                            piece_type = "silver"
                        elif next_char == 'b':
                            piece_type = "bishop"
                        elif next_char == 'r':
                            piece_type = "rook"
                        col_idx += 1  # +記号の次の文字をスキップ
                        
                    self.board[row_idx][col_idx] = ShogiPiece(piece_type, player, promoted)
                    col_idx += 1
        
        # 手番の設定
        self.current_player = Player.SENTE if turn == 'b' else Player.GOTE
        
        # 持ち駒の設定
        if captures != '-':
            i = 0
            while i < len(captures):
                char = captures[i]
                if char.isdigit():
                    count = int(char)
                    i += 1
                else:
                    count = 1
                
                piece_char = captures[i].lower()
                player = Player.SENTE if captures[i].isupper() else Player.GOTE
                
                if piece_char == 'p':
                    name = "pawn"
                elif piece_char == 'l':
                    name = "lance"
                elif piece_char == 'n':
                    name = "knight"
                elif piece_char == 's':
                    name = "silver"
                elif piece_char == 'g':
                    name = "gold"
                elif piece_char == 'b':
                    name = "bishop"
                elif piece_char == 'r':
                    name = "rook"
                
                self.captures[player][name] = self.captures[player].get(name, 0) + count
                i += 1
        
        # 局面の履歴に現在の局面を追加
        self.position_history.append(self.get_state_hash())
    
    def get_valid_moves(self) -> List[str]:
        """
        現在のプレイヤーの有効な手のリストを返す
        
        Returns:
            有効な手のリスト (USI形式の文字列)
        """
        moves = []
        
        # 盤上の駒の移動
        for row in range(9):
            for col in range(9):
                piece = self.board[row][col]
                if piece and piece.player == self.current_player:
                    for to_row, to_col in piece.get_moves((row, col), self.board):
                        # 通常の移動
                        move = self.coords_to_move((row, col), (to_row, to_col))
                        
                        # 成れる場合は、成る手と成らない手の両方を追加
                        if piece.can_promote_at_position(((row, col), (to_row, to_col))):
                            if not piece.must_promote(((row, col), (to_row, to_col))):
                                moves.append(move)
                            moves.append(move + "+")
                        else:
                            moves.append(move)
        
        # 持ち駒の打ち込み
        for piece_name, count in self.captures[self.current_player].items():
            if count > 0:
                for row in range(9):
                    for col in range(9):
                        if self.board[row][col] is None:
                            # 歩と香車と桂馬は、特定の段には打てない
                            if piece_name == "pawn" or piece_name == "lance":
                                if (self.current_player == Player.SENTE and row == 0) or (self.current_player == Player.GOTE and row == 8):
                                    continue
                            if piece_name == "knight":
                                if (self.current_player == Player.SENTE and row <= 1) or (self.current_player == Player.GOTE and row >= 7):
                                    continue
                                    
                            # 二歩の判定（同じ筋に自分の歩がある場合は打てない）
                            if piece_name == "pawn":
                                has_pawn = False
                                for r in range(9):
                                    p = self.board[r][col]
                                    if p and p.name == "pawn" and p.player == self.current_player and not p.promoted:
                                        has_pawn = True
                                        break
                                if has_pawn:
                                    continue
                                    
                            # 打ち歩詰めの判定
                            if piece_name == "pawn":
                                # 一時的に歩を打ってみる
                                self.board[row][col] = ShogiPiece("pawn", self.current_player)
                                self.captures[self.current_player][piece_name] -= 1
                                
                                # 詰みになるかチェック
                                is_mate = self.is_checkmate()
                                
                                # 元に戻す
                                self.board[row][col] = None
                                self.captures[self.current_player][piece_name] += 1
                                
                                if is_mate:
                                    continue  # 打ち歩詰めは禁止
                            
                            # 持ち駒を打つ手を追加
                            move = f"{piece_name[0].upper()}*{self.coords_to_square(row, col)}"
                            moves.append(move)
        
        return moves
    
    def coords_to_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """
        座標をUSI形式の指し手に変換
        
        Args:
            from_pos: 移動元の座標 (row, col)
            to_pos: 移動先の座標 (row, col)
            
        Returns:
            USI形式の指し手（例: "7g7f"）
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        return f"{self.coords_to_square(from_row, from_col)}{self.coords_to_square(to_row, to_col)}"
    
    def coords_to_square(self, row: int, col: int) -> str:
        """
        座標をUSI形式のマス目表記に変換
        
        Args:
            row: 行
            col: 列
            
        Returns:
            USI形式のマス目表記（例: "7g"）
        """
        file = 9 - col  # 筋（左から右へ 9～1）
        rank = chr(ord('a') + row)  # 段（上から下へ a～i）
        return f"{file}{rank}"
    
    def square_to_coords(self, square: str) -> Tuple[int, int]:
        """
        USI形式のマス目表記を座標に変換
        
        Args:
            square: USI形式のマス目表記（例: "7g"）
            
        Returns:
            座標 (row, col)
        """
        file = int(square[0])
        rank = square[1]
        col = 9 - file
        row = ord(rank) - ord('a')
        return (row, col)
    
    def move(self, move_str: str) -> bool:
        """
        指し手を実行する
        
        Args:
            move_str: USI形式の指し手（例: "7g7f", "7g7f+", "S*5e"）
            
        Returns:
            手が有効かどうか
        """
        # 有効な手のリストを取得
        valid_moves = self.get_valid_moves()
        if move_str not in valid_moves:
            return False
        
        # 持ち駒を打つ手
        if '*' in move_str:
            piece_type_char = move_str[0].lower()
            to_square = move_str[2:]
            to_row, to_col = self.square_to_coords(to_square)
            
            # 駒の種類を特定
            if piece_type_char == 'p':
                piece_type = "pawn"
                piece_name = "pawn"
            elif piece_type_char == 'l':
                piece_type = "lance"
                piece_name = "lance"
            elif piece_type_char == 'n':
                piece_type = "knight"
                piece_name = "knight"
            elif piece_type_char == 's':
                piece_type = "silver"
                piece_name = "silver"
            elif piece_type_char == 'g':
                piece_type = "gold"
                piece_name = "gold"
            elif piece_type_char == 'b':
                piece_type = "bishop"
                piece_name = "bishop"
            elif piece_type_char == 'r':
                piece_type = "rook"
                piece_name = "rook"
            
            # 持ち駒を減らす
            self.captures[self.current_player][piece_name] -= 1
            if self.captures[self.current_player][piece_name] == 0:
                del self.captures[self.current_player][piece_name]
            
            # 駒を盤上に配置
            self.board[to_row][to_col] = ShogiPiece(piece_type, self.current_player)
            
        # 駒を移動する手
        else:
            promotion = move_str.endswith('+')
            if promotion:
                move_str = move_str[:-1]
            
            from_square = move_str[:2]
            to_square = move_str[2:]
            
            from_row, from_col = self.square_to_coords(from_square)
            to_row, to_col = self.square_to_coords(to_square)
            
            piece = self.board[from_row][from_col]
            
            # 移動先に相手の駒があれば取る
            to_piece = self.board[to_row][to_col]
            if to_piece:
                captured_name = to_piece.name
                if to_piece.promoted:
                    # 成り駒は元の駒に戻す
                    captured_name = to_piece.base_piece_type
                
                # 持ち駒に追加
                if captured_name not in self.captures[self.current_player]:
                    self.captures[self.current_player][captured_name] = 0
                self.captures[self.current_player][captured_name] += 1
            
            # 駒を移動
            self.board[to_row][to_col] = piece
            self.board[from_row][from_col] = None
            
            # 成る場合
            if promotion:
                self.board[to_row][to_col] = piece.get_promoted_piece()
        
        # 手の履歴に追加
        self.move_history.append(move_str)
        
        # 手番を交代
        self.current_player = Player.GOTE if self.current_player == Player.SENTE else Player.SENTE
        
        # 局面の履歴に追加
        self.position_history.append(self.get_state_hash())
        
        return True
    
    def get_state_hash(self) -> str:
        """
        現在の盤面の状態をハッシュ値として取得
        
        Returns:
            盤面の状態を表すハッシュ文字列
        """
        state = []
        
        # 盤面の状態
        for row in range(9):
            for col in range(9):
                piece = self.board[row][col]
                if piece:
                    state.append(f"{row}{col}{piece.name}{piece.player.value}{piece.promoted}")
                else:
                    state.append(f"{row}{col}None")
        
        # 持ち駒の状態
        for player in [Player.SENTE, Player.GOTE]:
            for piece_name, count in sorted(self.captures[player].items()):
                state.append(f"c{player.value}{piece_name}{count}")
        
        # 手番
        state.append(f"turn{self.current_player.value}")
        
        # ハッシュ化
        return hashlib.md5("".join(state).encode()).hexdigest()
    
    def is_check(self) -> bool:
        """
        現在のプレイヤーの玉が王手されているかを判定
        
        Returns:
            王手されているかどうか
        """
        # 玉の位置を探す
        king_pos = None
        for row in range(9):
            for col in range(9):
                piece = self.board[row][col]
                if piece and piece.name == "king" and piece.player == self.current_player:
                    king_pos = (row, col)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return False  # 玉がない（通常はあり得ない）
        
        # 相手の駒からの攻撃があるかチェック
        for row in range(9):
            for col in range(9):
                piece = self.board[row][col]
                if piece and piece.player != self.current_player:
                    moves = piece.get_moves((row, col), self.board)
                    if king_pos in moves:
                        return True
        
        return False
    
    def is_checkmate(self) -> bool:
        """
        詰み判定
        
        Returns:
            詰みかどうか
        """
        # 王手されていない場合は詰みではない
        if not self.is_check():
            return False
        
        # 合法手が存在する場合は詰みではない
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return True  # 合法手がなければ詰み
            
        # 各合法手を試して王手が回避できるかチェック
        for move in valid_moves:
            # 一時的に手を実行
            original_board = [row[:] for row in self.board]
            original_captures = {
                Player.SENTE: self.captures[Player.SENTE].copy(),
                Player.GOTE: self.captures[Player.GOTE].copy()
            }
            original_player = self.current_player
            
            self.move(move)
            
            # 自分の手番に戻す
            self.current_player = original_player
            
            # 王手が回避できるかチェック
            is_still_check = self.is_check()
            
            # 元に戻す
            self.board = original_board
            self.captures = original_captures
            self.current_player = original_player
            
            if not is_still_check:
                return False  # 王手回避可能
        
        return True  # 詰み
    
    def is_repetition_draw(self) -> bool:
        """
        千日手の判定
        
        Returns:
            千日手かどうか
        """
        # 同一局面が4回以上出現したら千日手
        current_hash = self.position_history[-1]
        count = self.position_history.count(current_hash)
        return count >= 4
    
    def undo_last_move(self) -> bool:
        """
        最後の指し手を取り消す（テスト用）
        
        Returns:
            取り消しが成功したかどうか
        """
        if not self.move_history:
            return False
        
        # 盤面と持ち駒の履歴を1つ前に戻す処理（実装略）
        # 実際のアプリケーションでは、移動履歴に基づいて正確に元の状態に復元する必要がある
        
        # 履歴から削除
        self.move_history.pop()
        self.position_history.pop()
        
        return True
    
    def __str__(self) -> str:
        """盤面の文字列表現"""

        return BoardVisualizer.board_to_string(self.board, self.captures, self.current_player)