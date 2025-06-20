"""
モンテカルロ木探索（MCTS）の実装

このモジュールでは、強化学習におけるMCTS（Monte Carlo Tree Search）アルゴリズムを実装しています。
MCTSは、将棋のような複雑なゲームにおいて、状態空間を効率的に探索するための手法です。
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.shogi.shogi_pieces import (
    EMPTY, PAWN_S, LANCE_S, KNIGHT_S, SILVER_S, GOLD_S, BISHOP_S, ROOK_S, KING_S,
    PAWN_G, LANCE_G, KNIGHT_G, SILVER_G, GOLD_G, BISHOP_G, ROOK_G, KING_G,
    SENTE, GOTE, get_piece_owner, can_promote, get_base_piece, is_promoted,
    promoted_piece, piece_moves, long_move_pieces
)
from src.shogi.board_encoder import encode_board, visualize_board


@dataclass
class DiscreteActionConfig:
    """離散行動空間の基本設定"""
    action_num: int = 2187  # 行動空間の次元数（9x9の盤面 + 持ち駒の打ち先）


@dataclass
class MCTSConfig(DiscreteActionConfig):
    """MCTSのハイパーパラメータ設定"""
    simulation_times: int = 100     # 1回の探索で実行するシミュレーション回数
    expansion_threshold: int = 1     # ノードを展開する訪問回数の閾値
    gamma: float = 1.0               # 割引率
    uct_c: float = 1.5               # UCTの探索パラメータ
    dirichlet_alpha: float = 0.3     # ディリクレノイズのパラメータ
    dirichlet_weight: float = 0.25   # ディリクレノイズの重み
    value_weight: float = 0.5        # 価値と方策の重み付け（0.5で均等）


class RLParameter:
    """強化学習パラメータの基底クラス"""
    def __init__(self, config=None):
        self.config = config
        

class MCTSParameter(RLParameter):
    """MCTSの訪問回数と累積報酬を管理するクラス"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.N = {}  # 訪問回数 {state_key: [action_0_count, action_1_count, ...]}
        self.W = {}  # 累積報酬 {state_key: [action_0_reward, action_1_reward, ...]}
        self.Q = {}  # 平均報酬 {state_key: [action_0_value, action_1_value, ...]}
        self.P = {}  # 方策確率 {state_key: [action_0_prob, action_1_prob, ...]}
    
    def init_state(self, state_key, action_num, policy_probs=None):
        """新しい状態を初期化する"""
        if state_key not in self.N:
            self.N[state_key] = [0 for _ in range(action_num)]
            self.W[state_key] = [0.0 for _ in range(action_num)]
            self.Q[state_key] = [0.0 for _ in range(action_num)]
            
            # 方策確率を初期化（指定がなければ一様分布）
            if policy_probs is None:
                self.P[state_key] = [1.0 / action_num for _ in range(action_num)]
            else:
                self.P[state_key] = policy_probs
    
    def update(self, state_key, action, reward):
        """訪問回数と累積報酬を更新する"""
        self.N[state_key][action] += 1
        self.W[state_key][action] += reward
        self.Q[state_key][action] = self.W[state_key][action] / self.N[state_key][action]
    
    def get_uct_value(self, state_key, action, parent_n, c_puct):
        """UCT値を計算する"""
        # Q値（exploitation）
        q_value = self.Q[state_key][action]
        
        # P値と訪問回数に基づく探索項（exploration）
        n = self.N[state_key][action]
        p = self.P[state_key][action]
        exploration = c_puct * p * math.sqrt(parent_n) / (1 + n)
        
        return q_value + exploration
    
    def select_action(self, state_key, parent_n, c_puct, temperature=1.0):
        """UCT値に基づいてアクションを選択する"""
        uct_values = [
            self.get_uct_value(state_key, a, parent_n, c_puct)
            for a in range(len(self.N[state_key]))
        ]
        
        if temperature == 0:  # 最適行動を選択
            return np.argmax(uct_values)
        else:  # ソフトマックス選択
            exp_values = np.exp(np.array(uct_values) / temperature)
            probs = exp_values / np.sum(exp_values)
            return np.random.choice(len(probs), p=probs)
    
    def get_action_probs(self, state_key, temperature=1.0):
        """訪問回数に基づく行動確率を取得する"""
        counts = np.array(self.N[state_key])
        if temperature == 0:  # 決定論的に最大値を選択
            best_actions = np.where(counts == np.max(counts))[0]
            probs = np.zeros(len(counts))
            probs[best_actions] = 1.0 / len(best_actions)
        else:  # 温度付きソフトマックス
            counts = counts ** (1.0 / temperature)
            probs = counts / np.sum(counts)
        
        return probs


class MCTSNode:
    """モンテカルロ木探索におけるノードを表すクラス"""
    
    def __init__(self, prior: float = 0.0):
        """
        ノードの初期化
        
        Args:
            prior: このノードの事前確率（ニューラルネットワークによる予測値）
        """
        self.visit_count = 0  # このノードへの訪問回数
        self.value_sum = 0.0  # このノードの価値の合計
        self.prior = prior    # 事前確率
        self.children = {}    # 子ノード {action: MCTSNode}
        self.reward = 0.0     # このノードに到達したときの即時報酬
        self.expanded = False # このノードが展開されているかどうか
        
    @property
    def value(self) -> float:
        """このノードの平均価値を返す"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def expand(self, actions: List[Tuple], action_probs: Dict[Tuple, float], reward: float = 0.0):
        """
        ノードを展開する（可能な行動とその確率を設定）
        
        Args:
            actions: 可能な行動のリスト
            action_probs: 行動とその確率の辞書
            reward: このノードに到達したときの即時報酬
        """
        self.expanded = True
        self.reward = reward
        
        for action in actions:
            # 行動確率が0の場合はスキップ
            if action not in action_probs or action_probs[action] == 0:
                continue
            # 子ノードを作成（行動確率を事前確率として設定）
            self.children[action] = MCTSNode(prior=action_probs[action])
    
    def select_child(self, c_puct: float) -> Tuple[Any, 'MCTSNode']:
        """
        UCB値に基づいて子ノードを選択する
        
        Args:
            c_puct: 探索と活用のバランスを取るためのハイパーパラメータ
            
        Returns:
            (行動, 選択された子ノード)
        """
        # 訪問回数の平方根（exploration項に使用）
        sqrt_sum_count = math.sqrt(self.visit_count)
        
        # UCB値が最大の行動を選択
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # UCB値の計算
            # Q値（活用）+ C_PUCT × P(s,a) × √N(s) / (1 + N(s,a))
            # Q値は子ノードから見た自分の価値なので、-child.valueとなる（ゲームは交互に手番）
            exploit = -child.value  # 負の値は相手のノードから見た場合の逆転
            explore = c_puct * child.prior * sqrt_sum_count / (1 + child.visit_count)
            score = exploit + explore
            
            # 最大スコアを更新
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def update(self, value: float):
        """
        ノードの統計情報を更新する
        
        Args:
            value: バックアップする価値
        """
        self.visit_count += 1
        self.value_sum += value


class RLTrainer:
    """強化学習のトレーナー基底クラス"""
    def __init__(self, parameter, remote_memory=None):
        self.parameter = parameter
        self.remote_memory = remote_memory
        self.train_count = 0
        
    def train(self):
        """モデルを訓練する"""
        raise NotImplementedError("サブクラスで実装する必要があります")


class MCTS:
    """モンテカルロ木探索アルゴリズムの実装"""
    
    def __init__(self, game, actor_critic, c_puct=1.0, n_simulations=400, dirichlet_alpha=0.3, 
                exploration_fraction=0.25, pb_c_init=1.25, pb_c_base=19652):
        """
        MCTSの初期化
        
        Args:
            game: 将棋ゲームのインスタンス
            actor_critic: 方策と価値を予測するニューラルネットワーク
            c_puct: 探索と活用のバランスを取るためのハイパーパラメータ
            n_simulations: 1回の探索で実行するシミュレーション回数
            dirichlet_alpha: ディリクレノイズのアルファパラメータ
            exploration_fraction: ルートノードにおけるエクスプロレーションノイズの割合
            pb_c_init: PUCT計算のための初期係数
            pb_c_base: PUCT計算のためのベース値
        """
        self.game = game
        self.actor_critic = actor_critic
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.pb_c_init = pb_c_init
        self.pb_c_base = pb_c_base
        
        # ルートノードの初期化
        self.root = MCTSNode()
        
        # ノードごとの統計情報を記録
        self.node_stats = defaultdict(lambda: {"visits": 0, "value": 0.0})
    
    def search(self):
        """
        1回のMCTS探索を実行する
        
        MCTSの4ステップ（選択、拡張、シミュレーション、バックアップ）を実行する
        """
        # ゲーム状態のコピーを作成
        game_copy = self.game.clone()
        
        # 現在のノード
        node = self.root
        
        # 選択フェーズ（ルートから葉ノードまで探索）
        action_history = []
        
        # 未展開のノードに到達するまで選択を繰り返す
        while node.expanded:
            # UCB値に基づいて次の行動と子ノードを選択
            action, node = node.select_child(self.c_puct)
            
            # 選択した行動を実行
            game_copy.move(action)
            action_history.append(action)
            
            # 終了状態に到達した場合は評価
            if game_copy.is_terminal():
                # 最終状態の結果を取得
                value = game_copy.get_result(game_copy.current_player)
                
                # バックアップ
                self._backpropagate(action_history, value)
                return
        
        # 拡張とシミュレーションフェーズ
        
        # 現在の状態の特徴量を取得
        state_features = game_copy.get_features()
        
        # 方策と価値を予測
        action_probs, value = self.actor_critic.predict(state_features)
        
        # 合法手のみに制限
        valid_moves = game_copy.get_valid_moves()
        valid_action_probs = {}
        for move in valid_moves:
            if move in action_probs:
                valid_action_probs[move] = action_probs[move]
        
        # 確率の正規化（合計が1になるように）
        if valid_action_probs:
            prob_sum = sum(valid_action_probs.values())
            if prob_sum > 0:
                valid_action_probs = {move: prob / prob_sum for move, prob in valid_action_probs.items()}
            else:
                # 全ての確率が0の場合、一様分布として扱う
                valid_action_probs = {move: 1.0 / len(valid_moves) for move in valid_moves}
        
        # 現在のノードを拡張
        node.expand(valid_moves, valid_action_probs)
        
        # バックアップフェーズ
        self._backpropagate(action_history, value)
    
    def _backpropagate(self, action_history: List, value: float):
        """
        バックアップフェーズの実装
        
        Args:
            action_history: 訪れた行動の履歴
            value: バックアップする価値
        """
        # ルートノードから始まる現在のゲーム状態をコピー
        game_copy = self.game.clone()
        
        # 終了状態から現在のプレイヤーから見た価値
        current_value = value
        
        # バックアップ対象としてルートノードを追加
        nodes_to_update = [self.root]
        
        # 行動履歴に従って状態を進める
        for i, action in enumerate(action_history):
            # 行動を実行
            game_copy.move(action)
            
            # 対応するノードを取得
            node = nodes_to_update[-1].children[action]
            nodes_to_update.append(node)
            
            # 状態のハッシュを計算（統計情報用）
            state_hash = game_copy.get_state_hash()
            
            # 統計情報を更新
            self.node_stats[state_hash]["visits"] += 1
            self.node_stats[state_hash]["value"] = (
                self.node_stats[state_hash]["value"] * (self.node_stats[state_hash]["visits"] - 1) + current_value
            ) / self.node_stats[state_hash]["visits"]
        
        # 各ノードの値を更新
        for i, node in enumerate(nodes_to_update):
            # 奇数インデックスは相手の手番なので、価値の符号を反転
            if i % 2 == 1:
                node.update(-current_value)
            else:
                node.update(current_value)
    
    def get_action_probabilities(self, temperature=1.0):
        """
        各行動の確率分布を計算する
        
        Args:
            temperature: 温度パラメータ（低いほど確率が高い行動に集中）
            
        Returns:
            (選択された行動, {行動: 確率})
        """
        visit_counts = {action: child.visit_count for action, child in self.root.children.items()}
        
        if not visit_counts:
            # 子ノードがない場合（初期状態など）
            valid_moves = self.game.get_valid_moves()
            return np.random.choice(valid_moves), {move: 1.0 / len(valid_moves) for move in valid_moves}
            
        # 温度パラメータに基づいて確率分布を調整
        if temperature == 0:  # 決定的な選択
            action = max(visit_counts.items(), key=lambda x: x[1])[0]
            probs = {action: 1.0}
            return action, probs
        else:
            # 訪問回数を確率に変換
            counts = np.array([count for count in visit_counts.values()])
            if temperature != 1.0:
                # 温度で調整（低温ほど最大値に確率が集中）
                counts = counts ** (1.0 / temperature)
            # 正規化
            total = counts.sum()
            if total == 0:
                # すべての訪問回数が0の場合
                probs = {action: 1.0 / len(visit_counts) for action in visit_counts.keys()}
            else:
                probs = {action: count / total for action, count in zip(visit_counts.keys(), counts)}
            
            # 確率に基づいてランダムに行動を選択
            actions = list(probs.keys())
            probabilities = list(probs.values())
            action = np.random.choice(actions, p=probabilities)
            
            return action, probs
    
    def update_with_move(self, action):
        """
        指定された行動を実行し、探索木を更新する
        
        Args:
            action: 実行する行動
            
        Returns:
            行動が有効かどうか
        """
        if action in self.root.children:
            # 選択された子ノードを新しいルートに設定
            self.root = self.root.children[action]
            self.root.parent = None  # 親への参照を切る
            return True
        else:
            # 指定された行動が存在しない場合
            # 新しいゲーム状態でルートを初期化
            self.root = MCTSNode(0)
            return False
    
    def add_exploration_noise(self):
        """ルートノードに探索ノイズを追加"""
        if not self.root.expanded:
            return
            
        # ディリクレノイズを生成
        actions = list(self.root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        
        # 探索ノイズを追加
        for i, action in enumerate(actions):
            child = self.root.children[action]
            child.prior = child.prior * (1 - self.exploration_fraction) + noise[i] * self.exploration_fraction


class MCTSTrainer(RLTrainer):
    """MCTSのトレーナー"""
    def train(self):
        """メモリーに経験が来たらパラメータを更新する"""
        batchs = self.remote_memory.sample()
        for batch in batchs:
            state = batch["state"]
            action = batch["action"]
            reward = batch["reward"]
            self.parameter.init_state(state)

            self.parameter.N[state][action] += 1
            self.parameter.W[state][action] += reward
            self.train_count += 1
        return {}


def test_mcts():
    """MCTSのテスト"""
    from model.swin_shogi import SwinShogiModel
    
    # シード固定
    key = jax.random.PRNGKey(42)
    
    # モデルの初期化
    model = SwinShogiModel(
        embed_dim=96,
        depths=(2, 2),
        num_heads=(8, 16),
        window_size=3,
        mlp_ratio=4.0,
        dropout=0.0,
        policy_dim=2187
    )
    
    # 初期盤面を作成
    board, hands, turn = create_initial_board()
    print("初期盤面:")
    print(visualize_board(board, hands, turn))
    
    # 盤面をエンコード（モデル初期化用）
    features = encode_board(board, hands, turn)
    features = jnp.expand_dims(features, axis=0)  # バッチ次元を追加
    
    # パラメータ初期化
    params = model.init(key, features)
    
    # MCTSの設定
    config = MCTSConfig(
        simulation_times=10,      # シミュレーション回数
        expansion_threshold=1,    # 展開閾値
        gamma=1.0,                # 割引率
        uct_c=1.5,                # UCTの探索パラメータ
        dirichlet_alpha=0.3,      # ディリクレノイズのパラメータ
        dirichlet_weight=0.25     # ディリクレノイズの重み
    )
    
    # MCTSを初期化
    mcts = MCTS(model, params, config)
    
    # 探索を実行
    print("MCTS探索を実行中...")
    start_time = time.time()
    best_move, visit_counts, action_probs = mcts.search()
    end_time = time.time()
    
    print(f"探索時間: {end_time - start_time:.2f}秒")
    print(f"最適な手: {best_move}")
    print(f"訪問回数: {visit_counts}")
    print(f"行動確率: {action_probs}")
    
    # 手を実行
    new_board, new_hands, new_turn = mcts._apply_move(best_move)
    
    print("\n手を実行後の盤面:")
    print(visualize_board(new_board, new_hands, new_turn))
    
    return mcts, best_move


if __name__ == "__main__":
    test_mcts() 