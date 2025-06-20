"""将棋の盤面をニューラルネットワーク入力用にエンコード"""

import numpy as np
from .shogi_pieces import *

def encode_board(state):
    """
    将棋の状態をニューラルネットワークの入力形式にエンコード
    
    Args:
        state: 将棋環境の状態 (盤面、持ち駒、手番を含む)
    
    Returns:
        numpy.ndarray: エンコードされた状態 (チャネル, 高さ, 幅)
    """
    return encode_board_state(state)

def encode_board_state(state, history_length=8):
    """
    将棋の状態をニューラルネットワークの入力形式にエンコード
    
    Args:
        state: 将棋環境の状態 (盤面、持ち駒、手番を含む)
        history_length: 履歴として含める過去の状態数
    
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
            if piece != EMPTY:
                # 先手の駒
                if 1 <= piece <= 14:
                    planes[piece - 1][row][col] = 1
                # 後手の駒
                elif 15 <= piece <= 28:
                    planes[14 + (piece - 15)][row][col] = 1
    
    # 持ち駒をエンコード
    if SENTE in hands:
        for piece, count in hands[SENTE].items():
            if 1 <= piece <= 14:
                piece_idx = piece - 1
                # 持ち駒は最大で19枚と仮定
                count_capped = min(count, 19)
                planes[28 + piece_idx][0][0] = count_capped / 19.0  # 正規化
    
    if GOTE in hands:
        for piece, count in hands[GOTE].items():
            if 15 <= piece <= 28:
                piece_idx = piece - 15
                # 持ち駒は最大で19枚と仮定
                count_capped = min(count, 19)
                planes[28 + 14 + piece_idx][0][0] = count_capped / 19.0  # 正規化
    
    # 手番をエンコード
    if turn == SENTE:
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
    # 9x9の空の盤面を作成
    board = [[EMPTY for _ in range(9)] for _ in range(9)]
    
    # 後手の駒を配置（上側）
    board[0][0] = LANCE_G
    board[0][1] = KNIGHT_G
    board[0][2] = SILVER_G
    board[0][3] = GOLD_G
    board[0][4] = KING_G
    board[0][5] = GOLD_G
    board[0][6] = SILVER_G
    board[0][7] = KNIGHT_G
    board[0][8] = LANCE_G
    board[1][1] = BISHOP_G
    board[1][7] = ROOK_G
    for i in range(9):
        board[2][i] = PAWN_G
    
    # 先手の駒を配置（下側）
    board[8][0] = LANCE_S
    board[8][1] = KNIGHT_S
    board[8][2] = SILVER_S
    board[8][3] = GOLD_S
    board[8][4] = KING_S
    board[8][5] = GOLD_S
    board[8][6] = SILVER_S
    board[8][7] = KNIGHT_S
    board[8][8] = LANCE_S
    board[7][1] = ROOK_S
    board[7][7] = BISHOP_S
    for i in range(9):
        board[6][i] = PAWN_S
    
    # 初期状態には持ち駒はない
    hands = {
        SENTE: {},
        GOTE: {}
    }
    
    # 初期手番は先手
    turn = SENTE
    
    return {
        'board': board,
        'hands': hands,
        'turn': turn
    }

def visualize_board(state):
    """
    将棋の盤面をコンソールに表示する
    
    Args:
        state: 将棋環境の状態 (盤面、持ち駒、手番を含む)
    """
    board = state['board']
    hands = state['hands']
    turn = state['turn']
    
    # 駒の文字表現
    piece_symbols = {
        EMPTY: '・',
        # 先手の駒
        PAWN_S: '歩', LANCE_S: '香', KNIGHT_S: '桂', SILVER_S: '銀',
        GOLD_S: '金', BISHOP_S: '角', ROOK_S: '飛', KING_S: '玉',
        PROMOTED_PAWN_S: 'と', PROMOTED_LANCE_S: '杏', PROMOTED_KNIGHT_S: '圭', PROMOTED_SILVER_S: '全',
        PROMOTED_BISHOP_S: '馬', PROMOTED_ROOK_S: '龍',
        # 後手の駒
        PAWN_G: '↓歩', LANCE_G: '↓香', KNIGHT_G: '↓桂', SILVER_G: '↓銀',
        GOLD_G: '↓金', BISHOP_G: '↓角', ROOK_G: '↓飛', KING_G: '↓玉',
        PROMOTED_PAWN_G: '↓と', PROMOTED_LANCE_G: '↓杏', PROMOTED_KNIGHT_G: '↓圭', PROMOTED_SILVER_G: '↓全',
        PROMOTED_BISHOP_G: '↓馬', PROMOTED_ROOK_G: '↓龍'
    }

    # 行番号を漢字で表現
    row_kanji = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
    
    # 列のラベル
    print('   9    8    7   6    5    4    3    2    1  ')
    print('+----+----+----+----+----+----+----+----+----+')
    
    # 盤面表示
    for row in range(9):
        row_str = '|'
        for col in range(9):
            piece = board[row][col]
            symbol = piece_symbols.get(piece, '??')
            
            # 後手の駒（矢印がある駒）はスペースなしで表示
            is_gote = piece >= PAWN_G if piece != EMPTY else False
            
            if is_gote:
                row_str += f'{symbol} |'
            else:
                # 先手と空白マスはスペースを入れて中央揃え
                row_str += f' {symbol} |'
                
        print(row_str + f' {row_kanji[row]}')
        print('+----+----+----+----+----+----+----+----+----+')
    
    # 手番表示
    print(f"手番: {'先手(下)' if turn == SENTE else '後手(上)'}")
    
    # 持ち駒表示
    print('持ち駒:')
    
    # 先手の持ち駒
    sente_hand_str = '先手: '
    if SENTE in hands:
        for piece, count in hands[SENTE].items():
            if count > 0:
                symbol = piece_symbols.get(piece, '??')
                sente_hand_str += f'{symbol}{count} '
    print(sente_hand_str)
    
    # 後手の持ち駒
    gote_hand_str = '後手: '
    if GOTE in hands:
        for piece, count in hands[GOTE].items():
            if count > 0:
                symbol = piece_symbols.get(piece, '??')
                gote_hand_str += f'{symbol}{count} '
    print(gote_hand_str)

def decode_policy_output(policy_output):
    """
    ニューラルネットワークの方策出力を指し手に変換
    
    Args:
        policy_output: ネットワークの出力 (行動の確率分布)
    
    Returns:
        dict: {指し手: 確率} の形式
    """
    # ここでは簡易的に局面表現に戻すだけの実装とする
    # 実際の変換処理はMCTS側で実装
    actions = {}
    
    # policy_outputは長さ11259のベクトル (9x9x139 - 移動元81マス x 移動先81マス x 成り2通り = 13122)
    # ただし実際には不正な指し手が多いため、有効な手は数百程度
    
    return actions

def encode_move(move, board_size=9):
    """
    指し手をモデル入力用にエンコード
    
    Args:
        move: 指し手情報 (from, to, promote等)
        board_size: 盤面のサイズ
    
    Returns:
        int: エンコードされた指し手のインデックス
    """
    if "from" in move and move["from"] is not None:
        # 駒の移動
        from_pos = move["from"]
        to_pos = move["to"]
        promote = move.get("promote", False)
        
        from_idx = from_pos[0] * board_size + from_pos[1]
        to_idx = to_pos[0] * board_size + to_pos[1]
        
        # インデックスの計算
        # 移動元(81) x 移動先(81) x 成り(2)
        idx = from_idx * board_size * board_size * 2 + to_idx * 2 + (1 if promote else 0)
    else:
        # 駒打ち
        to_pos = move["to"]
        piece = move["piece"]
        
        # 持ち駒のタイプ (0-13: 先手、14-27: 後手)
        if 1 <= piece <= 14:
            piece_type = piece - 1
        else:
            piece_type = piece - 15 + 14
        
        to_idx = to_pos[0] * board_size + to_pos[1]
        
        # インデックスの計算
        # 81x81x2 (駒の移動分) + 駒種(28) x 移動先(81)
        idx = board_size * board_size * board_size * 2 + piece_type * board_size * board_size + to_idx
    
    return idx

def decode_move(idx, board_size=9):
    """
    エンコードされた指し手を元に戻す
    
    Args:
        idx: エンコードされた指し手のインデックス
        board_size: 盤面のサイズ
    
    Returns:
        dict: 指し手情報 (from, to, promote等)
    """
    # 駒の移動かどうか判定
    move_size = board_size * board_size * board_size * 2
    if idx < move_size:
        # 駒の移動
        promote = idx % 2 == 1
        tmp = idx // 2
        to_idx = tmp % (board_size * board_size)
        from_idx = tmp // (board_size * board_size)
        
        from_row = from_idx // board_size
        from_col = from_idx % board_size
        to_row = to_idx // board_size
        to_col = to_idx % board_size
        
        return {
            "from": (from_row, from_col),
            "to": (to_row, to_col),
            "promote": promote
        }
    else:
        # 駒打ち
        idx -= move_size
        to_idx = idx % (board_size * board_size)
        piece_type = idx // (board_size * board_size)
        
        to_row = to_idx // board_size
        to_col = to_idx % board_size
        
        # 駒のタイプを戻す
        if piece_type < 14:
            piece = piece_type + 1  # 先手
        else:
            piece = piece_type - 14 + 15  # 後手
        
        return {
            "to": (to_row, to_col),
            "piece": piece
        }