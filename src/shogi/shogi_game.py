"""
将棋のゲームロジックを実装するモジュール
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import hashlib
from src.shogi.shogi_pieces import ShogiPiece, Player
from src.shogi.board_visualizer import BoardVisualizer

@dataclass
class GameState:
    board: List[List[Optional[ShogiPiece]]]
    captures: Dict[Player, Dict[str, int]]
    current_player: Player
    move_history: List[str] = field(default_factory=list)
    position_history: List[str] = field(default_factory=list)
    is_terminal_flag: bool = False
    winner: Optional[Player] = None

    def to_dict(self) -> Dict[str, Any]:
        # This is needed for the evaluator in self_play
        return {
            "board": self.board,
            "hands": self.captures,
            "turn": self.current_player,
            "is_terminal": self.is_terminal_flag,
        }

class ShogiGame:
    """将棋のゲームを表すクラス"""
    
    def __init__(self, initial_state: Optional[GameState] = None):
        """ゲームの初期化"""
        if initial_state:
            self.game_state = initial_state
        else:
            self.game_state = self.get_initial_state()
        
    @staticmethod
    def get_initial_state() -> GameState:
        """初期配置のGameStateを返す"""
        board = [[None for _ in range(9)] for _ in range(9)]
        captures = {Player.SENTE: {}, Player.GOTE: {}}
        
        # 先手の駒配置（下側）
        board[8][0] = ShogiPiece("lance", Player.SENTE)
        board[8][1] = ShogiPiece("knight", Player.SENTE)
        board[8][2] = ShogiPiece("silver", Player.SENTE)
        board[8][3] = ShogiPiece("gold", Player.SENTE)
        board[8][4] = ShogiPiece("king", Player.SENTE)
        board[8][5] = ShogiPiece("gold", Player.SENTE)
        board[8][6] = ShogiPiece("silver", Player.SENTE)
        board[8][7] = ShogiPiece("knight", Player.SENTE)
        board[8][8] = ShogiPiece("lance", Player.SENTE)
        board[7][1] = ShogiPiece("bishop", Player.SENTE)
        board[7][7] = ShogiPiece("rook", Player.SENTE)
        for col in range(9):
            board[6][col] = ShogiPiece("pawn", Player.SENTE)
            
        # 後手の駒配置（上側）
        board[0][0] = ShogiPiece("lance", Player.GOTE)
        board[0][1] = ShogiPiece("knight", Player.GOTE)
        board[0][2] = ShogiPiece("silver", Player.GOTE)
        board[0][3] = ShogiPiece("gold", Player.GOTE)
        board[0][4] = ShogiPiece("king", Player.GOTE)
        board[0][5] = ShogiPiece("gold", Player.GOTE)
        board[0][6] = ShogiPiece("silver", Player.GOTE)
        board[0][7] = ShogiPiece("knight", Player.GOTE)
        board[0][8] = ShogiPiece("lance", Player.GOTE)
        board[1][1] = ShogiPiece("rook", Player.GOTE)
        board[1][7] = ShogiPiece("bishop", Player.GOTE)
        for col in range(9):
            board[2][col] = ShogiPiece("pawn", Player.GOTE)
        
        initial_game_state = GameState(board=board, captures=captures, current_player=Player.SENTE)
        initial_game_state.position_history.append(ShogiGame.get_state_hash(initial_game_state))
        return initial_game_state

    @staticmethod
    def get_valid_moves(game_state: GameState) -> List[str]:
        """
        現在のプレイヤーの有効な手のリストを返す
        """
        moves = []
        
        # 盤上の駒の移動
        for row in range(9):
            for col in range(9):
                piece = game_state.board[row][col]
                if piece and piece.player == game_state.current_player:
                    for to_row, to_col in piece.get_moves((row, col), game_state.board):
                        move = ShogiGame.coords_to_move((row, col), (to_row, to_col))
                        
                        if piece.can_promote_at_position(((row, col), (to_row, to_col))):
                            if not piece.must_promote(((row, col), (to_row, to_col))):
                                moves.append(move)
                            moves.append(move + "+")
                        else:
                            moves.append(move)
        
        # 持ち駒の打ち込み
        for piece_name, count in game_state.captures[game_state.current_player].items():
            if count > 0:
                for row in range(9):
                    for col in range(9):
                        if game_state.board[row][col] is None:
                            if piece_name in ["pawn", "lance"] and ((game_state.current_player == Player.SENTE and row == 0) or (game_state.current_player == Player.GOTE and row == 8)):
                                continue
                            if piece_name == "knight" and ((game_state.current_player == Player.SENTE and row <= 1) or (game_state.current_player == Player.GOTE and row >= 7)):
                                continue
                            if piece_name == "pawn":
                                has_pawn = any(p and p.name == "pawn" and p.player == game_state.current_player and not p.promoted for p in [game_state.board[r][col] for r in range(9)])
                                if has_pawn:
                                    continue
                                
                                temp_board = [r[:] for r in game_state.board]
                                temp_board[row][col] = ShogiPiece("pawn", game_state.current_player)
                                temp_captures = {p: c.copy() for p, c in game_state.captures.items()}
                                temp_captures[game_state.current_player][piece_name] -= 1
                                temp_state = GameState(temp_board, temp_captures, game_state.current_player)
                                if ShogiGame.is_checkmate(temp_state):
                                    continue
                            
                            piece_char = ShogiGame._get_piece_char_from_type(piece_name)
                            if piece_char:
                                moves.append(f"{piece_char}*{ShogiGame.coords_to_square(row, col)}")
        
        return moves

    @staticmethod
    def apply_action(game_state: GameState, move_str: str) -> GameState:
        new_board = [row[:] for row in game_state.board]
        new_captures = {p: c.copy() for p, c in game_state.captures.items()}
        
        if '*' in move_str:
            piece_type_char = move_str[0]
            to_square = move_str[2:]
            to_row, to_col = ShogiGame.square_to_coords(to_square)
            piece_type = ShogiPiece.PIECE_CHAR_INFO[piece_type_char]["type"]
            
            new_captures[game_state.current_player][piece_type] -= 1
            if new_captures[game_state.current_player][piece_type] == 0:
                del new_captures[game_state.current_player][piece_type]
            
            new_board[to_row][to_col] = ShogiPiece(piece_type, game_state.current_player)
        else:
            promotion = move_str.endswith('+')
            if promotion:
                move_str = move_str[:-1]
            
            from_square, to_square = move_str[:2], move_str[2:]
            from_row, from_col = ShogiGame.square_to_coords(from_square)
            to_row, to_col = ShogiGame.square_to_coords(to_square)
            
            piece = new_board[from_row][from_col]
            to_piece = new_board[to_row][to_col]
            
            if to_piece:
                captured_name = to_piece.base_piece_type
                new_captures[game_state.current_player][captured_name] = new_captures[game_state.current_player].get(captured_name, 0) + 1
            
            new_board[to_row][to_col] = piece
            new_board[from_row][from_col] = None
            
            if promotion:
                new_board[to_row][to_col] = piece.get_promoted_piece()

        next_player = Player.GOTE if game_state.current_player == Player.SENTE else Player.SENTE
        new_move_history = game_state.move_history + [move_str]
        
        new_game_state = GameState(new_board, new_captures, next_player, new_move_history, game_state.position_history[:])
        new_game_state.position_history.append(ShogiGame.get_state_hash(new_game_state))
        
        is_terminal, winner = ShogiGame.is_terminal(new_game_state)
        new_game_state.is_terminal_flag = is_terminal
        new_game_state.winner = winner
        
        return new_game_state

    @staticmethod
    def is_terminal(game_state: GameState) -> Tuple[bool, Optional[Player]]:
        if ShogiGame.is_checkmate(game_state):
            return True, Player.GOTE if game_state.current_player == Player.SENTE else Player.SENTE
        if not ShogiGame.get_valid_moves(game_state):
            return True, None # Stalemate
        if ShogiGame.is_repetition_draw(game_state):
            return True, None # Draw
        return False, None

    @staticmethod
    def is_checkmate(game_state: GameState) -> bool:
        if not ShogiGame.is_check(game_state):
            return False
        
        original_player = game_state.current_player
        for move in ShogiGame.get_valid_moves(game_state):
            temp_state = ShogiGame.apply_action(game_state, move)
            # After apply_action, the player is switched. We need to check if the original player is still in check.
            temp_state.current_player = original_player
            if not ShogiGame.is_check(temp_state):
                return False
        return True

    @staticmethod
    def is_check(game_state: GameState) -> bool:
        king_pos = None
        for r in range(9):
            for c in range(9):
                p = game_state.board[r][c]
                if p and p.name == "king" and p.player == game_state.current_player:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        if not king_pos:
            return False

        opponent = Player.GOTE if game_state.current_player == Player.SENTE else Player.SENTE
        for r in range(9):
            for c in range(9):
                p = game_state.board[r][c]
                if p and p.player == opponent:
                    if king_pos in p.get_moves((r, c), game_state.board):
                        return True
        return False

    @staticmethod
    def is_repetition_draw(game_state: GameState) -> bool:
        current_hash = ShogiGame.get_state_hash(game_state)
        return game_state.position_history.count(current_hash) >= 4

    @staticmethod
    def get_state_hash(game_state: GameState) -> str:
        state_parts = []
        for r in range(9):
            for c in range(9):
                p = game_state.board[r][c]
                if p:
                    state_parts.append(f"{r}{c}{p.name}{p.player.value}{p.promoted}")
                else:
                    state_parts.append(f"{r}{c}N")
        
        for player in [Player.SENTE, Player.GOTE]:
            for piece_name, count in sorted(game_state.captures[player].items()):
                state_parts.append(f"c{player.value}{piece_name}{count}")
        
        state_parts.append(f"t{game_state.current_player.value}")
        return hashlib.md5("".join(state_parts).encode()).hexdigest()

    @staticmethod
    def _get_piece_char_from_type(piece_type: str) -> str:
        for char, info in ShogiPiece.PIECE_CHAR_INFO.items():
            if info["type"] == piece_type:
                return char
        return ""

    @staticmethod
    def coords_to_move(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        return f"{ShogiGame.coords_to_square(from_pos[0], from_pos[1])}{ShogiGame.coords_to_square(to_pos[0], to_pos[1])}"

    @staticmethod
    def coords_to_square(row: int, col: int) -> str:
        return f"{9 - col}{chr(ord('a') + row)}"

    @staticmethod
    def square_to_coords(square: str) -> Tuple[int, int]:
        return (ord(square[1]) - ord('a'), 9 - int(square[0]))

    def __str__(self) -> str:
        return BoardVisualizer.board_to_string(self.game_state.board, self.game_state.captures, self.game_state.current_player)
