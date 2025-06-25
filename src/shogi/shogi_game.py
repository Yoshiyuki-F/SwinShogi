"""
将棋のゲームロジックを実装するモジュール
"""
from src.shogi.shogi_pieces import (
    ShogiPiece, Player
)
from typing import List, Tuple, Dict
import hashlib
from src.shogi.board_visualizer import BoardVisualizer
    
class ShogiGame:
    """将棋のゲームを表すクラス"""
    
    # クラス変数として共有インスタンスを保持
    _shared_instance = None
    
    @classmethod
    def get_shared_instance(cls):
        """共有インスタンスを取得（なければ作成）"""
        if cls._shared_instance is None:
            cls._shared_instance = cls()
        return cls._shared_instance
    
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
            char_idx = 0
            while char_idx < len(row):
                char = row[char_idx]
                if char.isdigit():
                    col_idx += int(char)
                    char_idx += 1
                else:
                    player = Player.GOTE if char.islower() else Player.SENTE
                    promoted = False
                    
                    if char.lower() == '+':
                        # 成り駒の処理
                        promoted = True
                        char_idx += 1
                        next_char = row[char_idx].lower()
                        piece_char = f"+{next_char}"
                    else:
                        piece_char = char.lower()
                    
                    # PIECE_CHAR_INFOを使用して駒の種類を取得
                    if piece_char in ShogiPiece.PIECE_CHAR_INFO:
                        piece_type = ShogiPiece.PIECE_CHAR_INFO[piece_char]["type"]
                    else:
                        raise ValueError(f"不明な駒文字: {piece_char}")
                    
                    self.board[row_idx][col_idx] = ShogiPiece(piece_type, player, promoted)
                    col_idx += 1
                    char_idx += 1
        
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
                
                # PIECE_CHAR_INFOを使用して駒の種類を取得
                if piece_char in ShogiPiece.PIECE_CHAR_INFO:
                    name = ShogiPiece.PIECE_CHAR_INFO[piece_char]["type"]
                else:
                    raise ValueError(f"不明な持ち駒文字: {piece_char}")
                
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
                            # piece_nameからUSI文字を取得
                            piece_char = self._get_piece_char_from_type(piece_name)
                            if piece_char:
                                move = f"{piece_char}*{self.coords_to_square(row, col)}"
                                moves.append(move)
        
        return moves
    
    def _get_piece_char_from_type(self, piece_type: str) -> str:
        """駒の種類からUSI文字を取得"""
        for char, info in ShogiPiece.PIECE_CHAR_INFO.items():
            if info["type"] == piece_type:
                return char
        return ""
    
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
            piece_type_char = move_str[0]
            to_square = move_str[2:]
            to_row, to_col = self.square_to_coords(to_square)
            
            # 駒の種類を特定
            if piece_type_char in ShogiPiece.PIECE_CHAR_INFO:
                piece_type = ShogiPiece.PIECE_CHAR_INFO[piece_type_char]["type"]
                piece_name = piece_type  # 持ち駒の場合、piece_nameはpiece_typeと同じ
            else:
                return False
            
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

    
    def evaluate_with_model(self, model, params):
        """
        SwinTransformerで局面を評価する
        
        Args:
            model: SwinTransformerモデル
            params: モデルパラメータ

        Returns:
            評価値（正の値はプレイヤーが有利、負の値は不利）
        """

        player = self.current_player
            
        # 現在の局面をゲーム状態として作成
        game_state = {
            'board': [row[:] for row in self.board],  # Deep copy
            'hands': {k: v.copy() for k, v in self.captures.items()},  # Deep copy
            'turn': player
        }
        
        # SwinTransformerで評価
        from src.model.actor_critic import predict_for_mcts
        _, value = predict_for_mcts(model, params, game_state) #value is already a float
        
        # 詰みボーナスをチェック
        checkmate_bonus = self._calculate_checkmate_bonus(player)
        
        return value + checkmate_bonus
    
    @staticmethod
    def evaluate_batch(game_states, model, params):
        """
        複数の局面を同時に評価する（バッチ処理）
        
        Args:
            game_states: ゲーム状態のリスト
            model: SwinTransformerモデル  
            params: モデルパラメータ
            
        Returns:
            評価値のリスト
        """
        from src.model.actor_critic import predict_for_mcts
        
        values = []
        for game_state in game_states:
            _, value = predict_for_mcts(model, params, game_state)
            values.append(float(value))
            
        return values
    
    @property
    def game_state(self) -> Dict:
        """
        現在のゲーム状態を辞書として取得
        
        Returns:
            ゲーム状態の辞書（board, hands, turn, is_terminal, is_checkmate）
        """
        return {
            'board': [row[:] for row in self.board],  # Deep copy
            'hands': {k: v.copy() for k, v in self.captures.items()},  # Deep copy
            'turn': self.current_player,
            'is_terminal': self.is_terminal,
            'is_checkmate': self.is_checkmate()
        }
    
    @property
    def is_terminal(self) -> bool:
        """ゲームが終了状態かどうか"""
        return self.is_checkmate() or len(self.get_valid_moves()) == 0
    
    @property
    def terminal_value(self) -> float:
        """終了状態での価値を取得"""
        if not self.is_terminal:
            return 0.0
            
        if self.is_checkmate():
            # 現在のプレイヤーが詰まされている = 負け
            return -1.0
        else:
            # 引き分け
            return 0.0

    def _load_from_state(self, game_state: Dict):
        """
        ゲーム状態からインスタンスの状態を復元する
        
        Args:
            game_state: 復元するゲーム状態
        """
        self.board = [row[:] for row in game_state['board']]
        self.captures = {k: v.copy() for k, v in game_state['hands'].items()}
        self.current_player = game_state['turn']

    @classmethod
    def apply_action_to_state(cls, action: str, game_state: Dict) -> Dict:
        """
        指定されたゲーム状態にアクションを適用して新しい状態を返す
        
        Args:
            action: USI形式の手
            game_state: 適用対象のゲーム状態

        Returns:
            新しいゲーム状態
        """
        # 共有インスタンスを使用
        shared_game = cls.get_shared_instance()
        shared_game._load_from_state(game_state)
        
        # アクションを適用
        success = shared_game.move(action)
        if not success:
            raise ValueError(f"Invalid move: {action}")
        
        # 新しい状態を返す
        return shared_game.game_state
    
    @classmethod
    def get_valid_moves_from_state(cls, game_state: Dict) -> List[str]:
        """
        ゲーム状態から有効手のリストを取得
        
        Args:
            game_state: ゲーム状態
            
        Returns:
            有効手のリスト
        """
        # 共有インスタンスを使用
        shared_game = cls.get_shared_instance()
        shared_game._load_from_state(game_state)
        return shared_game.get_valid_moves()

    
    def _calculate_checkmate_bonus(self, player): #TODO かぶってる（詰みの場合のボーナスがどこででも定義されてるConfigに追加して、統一しよう。
        """詰みボーナスを計算"""
        # 現在の手番を一時保存
        original_player = self.current_player
        
        bonus = 0.0
        
        # 相手が詰んでいるかチェック
        opponent = Player.GOTE if player == Player.SENTE else Player.SENTE
        self.current_player = opponent
        if self.is_checkmate():
            bonus += 1000.0  # 相手を詰ませている場合は大きなプラス
            
        # 自分が詰んでいるかチェック  
        self.current_player = player
        if self.is_checkmate():
            bonus -= 1000.0  # 自分が詰まされている場合は大きなマイナス
            
        # 手番を復元
        self.current_player = original_player
        
        return bonus
    
    def __str__(self) -> str:
        """盤面の文字列表現"""

        return BoardVisualizer.board_to_string(self.board, self.captures, self.current_player)
