"""将棋の盤面をニューラルネットワーク入力用にエンコード"""

import numpy as np
from .shogi_pieces import Player, PieceType, ShogiPiece
from .board_visualizer import BoardVisualizer
from .shogi_game import ShogiGame

def encode_board_state(state):
    """
    将棋の状態をニューラルネットワークの入力形式にエンコード
    
    Args:
        state: 将棋環境の状態 (盤面、持ち駒、手番を含む)

    Returns:
        numpy.ndarray: エンコードされた状態 (チャネル, 高さ, 幅)
    """
    board = state['board']
    hands = state['hands']
    turn = state['turn']
    
    # 各駒の種類ごとに別々のプレーン
    # 先手14種(駒種) + 後手14種(駒種) + 持ち駒38種(先手19種+後手19種)
    # 持ち駒は0-18枚まで表現（実際はそこまで多くならない）
    # 手番 + 王手フラグ + 合法手マスク
    plane_count = 14 + 14 + 38 + 1 + 1 + 1
    planes = np.zeros((plane_count, 9, 9), dtype=np.float32)
    
    # 盤面の駒をエンコード
    for row in range(9):
        for col in range(9):
            piece = board[row][col]
            if piece is not None:
                piece_type_idx = piece.piece_type.value - 1
                if piece.player == Player.SENTE:
                    planes[piece_type_idx][row][col] = 1
                else:  # Player.GOTE
                    planes[14 + piece_type_idx][row][col] = 1
    
    # 持ち駒をエンコード
    if Player.SENTE in hands:
        for piece_name, count in hands[Player.SENTE].items():
            piece_type = BoardVisualizer.get_piece_type_from_name(piece_name)
            if piece_type is not None:
                piece_idx = piece_type.value - 1
                # 持ち駒は最大で19枚と仮定
                count_capped = min(count, 19)
                planes[28 + piece_idx][0][0] = count_capped / 19.0  # 正規化
    
    if Player.GOTE in hands:
        for piece_name, count in hands[Player.GOTE].items():
            piece_type = BoardVisualizer.get_piece_type_from_name(piece_name)
            if piece_type is not None:
                piece_idx = piece_type.value - 1
                # 持ち駒は最大で19枚と仮定
                count_capped = min(count, 19)
                planes[28 + 14 + piece_idx][0][0] = count_capped / 19.0  # 正規化
    
    # 手番をエンコード
    if turn == Player.SENTE:
        planes[28 + 38][:,:] = 1
    else:
        planes[28 + 38][:,:] = 0
    
    # 王手フラグ（ここでは判定できないため常に0）
    planes[28 + 38 + 1][:,:] = 0
    
    # 合法手マスク（ここでは実装せず、別途MCTSで生成）
    planes[28 + 38 + 2][:,:] = 0
    
    return planes

def create_initial_board():
    """
    将棋の初期盤面を生成
    
    Returns:
        dict: 初期状態の辞書 (盤面、持ち駒、手番を含む)
    """
    game = ShogiGame()  # 初期配置が設定されたゲームを作成
    
    return {
        'board': game.board,
        'hands': game.captures,
        'turn': game.current_player
    }


def encode_move(move, board_size=9):
    """
    USI形式の指し手をニューラルネットワーク用にエンコード
    
    Args:
        move: USI形式の指し手（例: "7g7f", "7g7f+", "S*5e"）
        board_size: 盤面のサイズ（通常は9x9）
        
    Returns:
        int: エンコードされた指し手のインデックス
    """
    # 移動元と移動先の座標を取得
    if '*' in move:  # 持ち駒を打つ手
        piece_type = move[0].lower()
        to_square = move[2:]
        to_file = int(to_square[0])
        to_rank = ord(to_square[1]) - ord('a')
        
        # 持ち駒を打つ手のインデックス計算
        # 駒の種類 * 81 + 移動先のマス
        piece_idx = {'p': 0, 'l': 1, 'n': 2, 's': 3, 'g': 4, 'b': 5, 'r': 6}[piece_type]
        return 2187 + piece_idx * 81 + (to_rank * board_size + (board_size - to_file))
    else:  # 盤上の駒を動かす手
        promotion = move.endswith('+')
        if promotion:
            move = move[:-1]
        
        from_square = move[:2]
        to_square = move[2:]
        
        from_file = int(from_square[0])
        from_rank = ord(from_square[1]) - ord('a')
        to_file = int(to_square[0])
        to_rank = ord(to_square[1]) - ord('a')
        
        # 移動元と移動先から指し手のインデックスを計算
        from_idx = from_rank * board_size + (board_size - from_file)
        to_idx = to_rank * board_size + (board_size - to_file)
        
        # 成る手は別のインデックス範囲を使用
        if promotion:
            return 1089 + from_idx * board_size + to_idx
        else:
            return from_idx * board_size + to_idx

def decode_move(idx, board_size=9):
    """
    エンコードされた指し手をUSI形式に戻す
    
    Args:
        idx: エンコードされた指し手のインデックス
        board_size: 盤面のサイズ（通常は9x9）
        
    Returns:
        str: USI形式の指し手
    """
    # 持ち駒を打つ手
    if idx >= 2187:
        drop_idx = idx - 2187
        piece_idx = drop_idx // 81
        to_pos = drop_idx % 81
        
        to_rank = to_pos // board_size
        to_file = board_size - (to_pos % board_size)
        
        piece_char = ['P', 'L', 'N', 'S', 'G', 'B', 'R'][piece_idx]
        return f"{piece_char}*{to_file}{chr(ord('a') + to_rank)}"
    
    # 成る手
    elif idx >= 1089:
        prom_idx = idx - 1089
        from_idx = prom_idx // board_size
        to_idx = prom_idx % board_size
        
        from_rank = from_idx // board_size
        from_file = board_size - (from_idx % board_size)
        to_rank = to_idx // board_size
        to_file = board_size - (to_idx % board_size)
        
        return f"{from_file}{chr(ord('a') + from_rank)}{to_file}{chr(ord('a') + to_rank)}+"
    
    # 通常の手
    else:
        from_idx = idx // board_size
        to_idx = idx % board_size
        
        from_rank = from_idx // board_size
        from_file = board_size - (from_idx % board_size)
        to_rank = to_idx // board_size
        to_file = board_size - (to_idx % board_size)
        
        return f"{from_file}{chr(ord('a') + from_rank)}{to_file}{chr(ord('a') + to_rank)}"