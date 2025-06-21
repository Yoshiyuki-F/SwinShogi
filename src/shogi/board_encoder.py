"""将棋の盤面をニューラルネットワーク入力用にエンコード"""

import numpy as np
from .shogi_pieces import Player, ShogiPiece

def encode_board_state(state):
    """
    将棋の状態をニューラルネットワークの入力形式にエンコード
    
    新しいエンコード仕様:
    - チャネル数: 2
        - チャネル0: 駒の種類 (0:駒なし, 1-14:駒の種類に対応する数値)
        - チャネル1: プレイヤー (0:駒なし, 1:先手, 2:後手)
    - 形状: (H, W, C) = (9, 9, 2)
    
    Args:
        state: 将棋環境の状態 (盤面、持ち駒、手番を含む)

    Returns:
        numpy.ndarray: エンコードされた状態 (高さ, 幅, チャネル)
    """
    board = state['board']
    
    # 2チャネルのエンコーディング
    # チャネル0: 駒の種類
    # チャネル1: プレイヤー
    # 形状: (H, W, C) = (9, 9, 2)
    planes = np.zeros((9, 9, 2), dtype=np.float32)
    
    # 盤面の駒をエンコード
    for row in range(9):
        for col in range(9):
            piece = board[row][col]
            if piece is not None:
                # 駒の種類をエンコード (1-14の値)
                piece_type_idx = 0  # デフォルト値
                for char, info in ShogiPiece.PIECE_CHAR_INFO.items():
                    if info["type"] == piece.name:
                        piece_type_idx = info["type_idx"]
                        break
                
                # チャネル0: 駒の種類 (1-indexed)
                planes[row, col, 0] = piece_type_idx + 1  # 0は駒なしを表すため+1
                
                # チャネル1: プレイヤー (1:先手, 2:後手)
                planes[row, col, 1] = 1 if piece.player == Player.SENTE else 2
    
    return planes

def get_feature_vector(state):
    """
    手番と持ち駒の情報を特徴ベクトルとして取得する
    
    特徴ベクトルの構成 (長さ15):
    - [0]: 手番 (0:先手, 1:後手)
    - [1-7]: 先手の持ち駒数 (type_index=0~6に対応)
    - [8-14]: 後手の持ち駒数 (type_index=0~6に対応)
    
    Args:
        state: 将棋環境の状態
        
    Returns:
        numpy.ndarray: 特徴ベクトル (長さ15)
    """
    hands = state['hands']
    turn = state['turn']
    
    # 15次元の特徴ベクトルを初期化
    feature_vector = np.zeros(15, dtype=np.float32)
    
    # [0]: 手番
    feature_vector[0] = 0 if turn == Player.SENTE else 1
    
    # [1-7]: 先手の持ち駒
    if Player.SENTE in hands:
        for piece_name, count in hands[Player.SENTE].items():
            # 駒の種類に対応するインデックスを取得
            for char, info in ShogiPiece.PIECE_CHAR_INFO.items():
                if info["type"] == piece_name and "type_idx" in info:
                    type_idx = info["type_idx"]
                    if 0 <= type_idx <= 6:  # 安全チェック
                        feature_vector[1 + type_idx] = count
                    break
    
    # [8-14]: 後手の持ち駒
    if Player.GOTE in hands:
        for piece_name, count in hands[Player.GOTE].items():
            # 駒の種類に対応するインデックスを取得
            for char, info in ShogiPiece.PIECE_CHAR_INFO.items():
                if info["type"] == piece_name and "type_idx" in info:
                    type_idx = info["type_idx"]
                    if 0 <= type_idx <= 6:  # 安全チェック
                        feature_vector[8 + type_idx] = count
                    break
    
    return feature_vector


def encode_move(move):
    """
    USI形式の指し手を9x9x139のアクションプレーンにエンコードする
    
    アクションプレーンの構成:
    - 0-63: クイーン移動（8方向 × 最大8マス）
    - 64-65: ナイト移動（2種類）
    - 66-129: 成りクイーン移動（8方向 × 最大8マス）
    - 130-131: 成りナイト移動（2種類）
    - 132-138: 持ち駒打ち（7種類の駒）
    
    Args:
        move: USI形式の指し手（例: "7g7f", "7g7f+", "S*5e"）
        
    Returns:
        tuple: (移動元の行, 移動元の列, アクションインデックス) 持ち駒の場合は(-1, -1, アクションインデックス)
    """
    # 持ち駒を打つ手
    if '*' in move:
        piece_type = move[0].lower()
        to_file = int(move[2])
        to_rank = ord(move[3]) - ord('a')
        
        # 駒の種類から持ち駒インデックスを取得 (132-138)
        piece_idx = 0
        for char, info in ShogiPiece.PIECE_CHAR_INFO.items():
            if char == piece_type and info.get("type_idx") is not None:
                piece_idx = info["type_idx"]
                break
        
        # 持ち駒のアクションインデックスは132から始まる
        action_idx = 132 + piece_idx
        
        # 持ち駒を打つ場合、移動元は特殊な値(-1, -1)とする
        # 仕様に合わせて移動元の行、列、アクションインデックスのみを返す
        return (-1, -1, action_idx)
    
    # 通常の手または成る手
    promotion = move.endswith('+')
    if promotion:
        move = move[:-1]
    
    from_file = int(move[0])
    from_rank = ord(move[1]) - ord('a')
    to_file = int(move[2])
    to_rank = ord(move[3]) - ord('a')
    
    # 移動元の座標（0-indexed）
    from_row = from_rank
    from_col = 9 - from_file
    
    # 移動先の座標（0-indexed）
    to_row = to_rank
    to_col = 9 - to_file
    
    # 移動方向と距離を計算
    d_row = to_row - from_row
    d_col = to_col - from_col
    
    # アクションインデックスを計算
    action_idx = -1
    
    # ナイト移動（桂馬）
    if (d_row == -2 and d_col == -1) or (d_row == -2 and d_col == 1):
        if d_col == -1:
            action_idx = 64  # 左桂馬
        else:
            action_idx = 65  # 右桂馬
        
        if promotion:
            action_idx += 66  # 成りナイト移動は130-131
    
    # クイーン移動（8方向）
    else:
        # 方向の正規化
        if d_row != 0:
            d_row_norm = d_row // abs(d_row)
        else:
            d_row_norm = 0
            
        if d_col != 0:
            d_col_norm = d_col // abs(d_col)
        else:
            d_col_norm = 0
        
        # 距離の計算
        if d_row_norm != 0 and d_col_norm != 0:
            # 斜め移動
            distance = abs(d_row)  # または abs(d_col)
        else:
            # 縦横移動
            distance = max(abs(d_row), abs(d_col))
        
        # 方向インデックス（0-7）
        dir_idx = -1
        if d_row_norm == -1 and d_col_norm == 0:  # 上
            dir_idx = 0
        elif d_row_norm == -1 and d_col_norm == 1:  # 右上
            dir_idx = 1
        elif d_row_norm == 0 and d_col_norm == 1:  # 右
            dir_idx = 2
        elif d_row_norm == 1 and d_col_norm == 1:  # 右下
            dir_idx = 3
        elif d_row_norm == 1 and d_col_norm == 0:  # 下
            dir_idx = 4
        elif d_row_norm == 1 and d_col_norm == -1:  # 左下
            dir_idx = 5
        elif d_row_norm == 0 and d_col_norm == -1:  # 左
            dir_idx = 6
        elif d_row_norm == -1 and d_col_norm == -1:  # 左上
            dir_idx = 7
        
        if dir_idx != -1:
            # クイーン移動のアクションインデックス（0-63）
            action_idx = dir_idx * 8 + (distance - 1)
            
            if promotion:
                # 成りクイーン移動のアクションインデックス（66-129）
                action_idx += 66
    
    # 仕様に合わせて移動元の行、列、アクションインデックスのみを返す
    return (from_row, from_col, action_idx)

def decode_move(from_row, from_col, action_idx):
    """
    アクションプレーンのインデックスからUSI形式の指し手に変換
    
    Args:
        from_row: 移動元の行（0-indexed）　持ち駒の場合は-1
        from_col: 移動元の列（0-indexed）　持ち駒の場合は-1
        action_idx: アクションインデックス（0-138）
        
    Returns:
        str: USI形式の指し手
    """
    #TODO: 持ち駒を打つ場合の処理を追加する
