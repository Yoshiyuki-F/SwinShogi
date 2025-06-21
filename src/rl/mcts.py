"""
モンテカルロ木探索（MCTS）の実装

このモジュールでは、強化学習におけるMCTS（Monte Carlo Tree Search）アルゴリズムを実装しています。
MCTSは、将棋のような複雑なゲームにおいて、状態空間を効率的に探索するための手法です。
"""

import math
import numpy as np
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.default_config import MCTS_CONFIG


@dataclass
class DiscreteActionConfig:
    """離散行動空間の基本設定"""
    action_num: int = MCTS_CONFIG['action_num']  # 行動空間の次元数（9x9の盤面 + 持ち駒の打ち先）


@dataclass
class MCTSConfig(DiscreteActionConfig):
    """MCTSのハイパーパラメータ設定"""
    simulation_times: int = MCTS_CONFIG['simulation_times']     # 1回の探索で実行するシミュレーション回数
    expansion_threshold: int = MCTS_CONFIG['expansion_threshold']     # ノードを展開する訪問回数の閾値
    gamma: float = MCTS_CONFIG['gamma']               # 割引率
    uct_c: float = MCTS_CONFIG['uct_c']               # UCTの探索パラメータ
    dirichlet_alpha: float = MCTS_CONFIG['dirichlet_alpha']     # ディリクレノイズのパラメータ
    dirichlet_weight: float = MCTS_CONFIG['dirichlet_weight']   # ディリクレノイズの重み
    value_weight: float = MCTS_CONFIG['value_weight']        # 価値と方策の重み付け（0.5で均等）

        

class MCTSParameter:
    """MCTSの訪問回数と累積報酬を管理するクラス"""
    
    def __init__(self):
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


class MCTS:
    """モンテカルロ木探索アルゴリズムの実装"""
    
    def __init__(self, game, actor_critic, c_puct=MCTS_CONFIG['uct_c'], n_simulations=MCTS_CONFIG['simulation_times'], 
                dirichlet_alpha=MCTS_CONFIG['dirichlet_alpha'], exploration_fraction=MCTS_CONFIG['dirichlet_weight'], 
                pb_c_init=MCTS_CONFIG['pb_c_init'], pb_c_base=MCTS_CONFIG['pb_c_base']):
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
        self.game_state = game.clone()  # 現在のゲーム状態
        
        # 探索開始時にルートノードを展開
        self._expand_root_node()
        
    def _expand_root_node(self):
        """ルートノードを展開"""
        # 現在の状態の特徴量を取得
        state_features = self.game_state.get_features()
        
        # 方策と価値を予測
        action_probs, _ = self.actor_critic.predict(state_features)
        
        # 有効な行動のみを残す
        valid_moves = self.game_state.get_valid_moves()
        valid_action_probs = {a: action_probs.get(a, 0) for a in valid_moves}
        
        # 確率の正規化（合計を1にする）
        if sum(valid_action_probs.values()) > 0:
            norm_factor = sum(valid_action_probs.values())
            valid_action_probs = {a: p / norm_factor for a, p in valid_action_probs.items()}
        else:
            # 全ての有効な行動に等しい確率を割り当てる
            valid_action_probs = {a: 1.0 / len(valid_moves) for a in valid_moves}
        
        # ルートノードを展開
        self.root.expand(valid_moves, valid_action_probs)

    def search(self):
        """
        モンテカルロ木探索を実行する
        
        シーケンス図の探索ループ部分に対応
        Returns:
            最適な行動とその確率の辞書
        """
        # 探索回数分だけシミュレーションを実行
        for i in range(self.n_simulations):
            # ゲーム状態をクローン
            sim_state = self.game_state.clone()
            
            # 選択、拡張、シミュレーション、バックプロパゲーション
            # シーケンス図のループに対応
            self._simulate_once(sim_state, self.root, 0)
            
        # 行動確率を計算
        action_probs = {}
        for action, child in self.root.children.items():
            action_probs[action] = child.visit_count / self.root.visit_count
            
        return action_probs
    
    def _simulate_once(self, state, node: MCTSNode, depth: int) -> float:
        """
        1回のシミュレーションを実行する
        
        Args:
            state: シミュレーション用のゲーム状態
            node: 現在のノード
            depth: 現在の深さ
            
        Returns:
            シミュレーション結果の価値（報酬）
        """
        # 終端状態の場合は評価値を返す
        if state.is_terminal():
            return state.get_reward()
            
        # ノードが展開されていない場合は展開
        if not node.expanded:
            # 新しいノードの評価要求（AC->Transformer->AC）
            # シーケンス図のMCTS->AC部分
            state_features = state.get_features()
            action_probs, value = self.actor_critic.predict(state_features)
            
            # 有効な行動のみを残す
            valid_moves = state.get_valid_moves()
            valid_action_probs = {a: action_probs.get(a, 0) for a in valid_moves}
            
            # 確率の正規化
            if sum(valid_action_probs.values()) > 0:
                norm_factor = sum(valid_action_probs.values())
                valid_action_probs = {a: p / norm_factor for a, p in valid_action_probs.items()}
            else:
                valid_action_probs = {a: 1.0 / len(valid_moves) for a in valid_moves}
            
            # ノードを展開
            node.expand(valid_moves, valid_action_probs)
            
            # 価値を返す（ニューラルネットワークからの予測値）
            return value
        
        # 子ノード選択（UCB値に基づく最良の行動を選択）
        action, child_node = node.select_child(self.c_puct)
        
        # 選択した行動を実行
        state.move(action)
        
        # 再帰的にシミュレーション（深さを1増やす）
        value = self._simulate_once(state, child_node, depth + 1)
        
        # バックプロパゲーション
        # シーケンス図のバックプロパゲーション部分
        node.update(-value)  # 価値を反転（相手側から見た価値なので）
        
        return -value  # 価値を反転して返す
    
    def _backpropagate(self, action_history: List, value: float):
        """
        バックプロパゲーション処理
        
        Args:
            action_history: 行動の履歴
            value: 最終的な価値
        """
        # こちらはシーケンス図には表示されていない詳細実装部分
        # 試合で価値を順方向に伝播する
        current_node = self.root
        current_player = self.game_state.current_player
        
        for action in action_history:
            # 現在のノードを更新
            current_node.update(value if current_player == self.game_state.current_player else -value)
            # 子ノードに移動
            if action in current_node.children:
                current_node = current_node.children[action]
            else:
                # 履歴に対応する子ノードがない場合は終了
                break
            # プレイヤーを交代
            current_player = 1 - current_player

    def get_action_probabilities(self, temperature=1.0):
        """
        探索結果から行動確率を計算する（最終的な行動選択用）
        
        Args:
            temperature: 温度パラメータ（0に近いほど最適解に確定的）
            
        Returns:
            行動とその確率の辞書
        """
        # 訪問回数を配列に変換
        visits = {action: child.visit_count for action, child in self.root.children.items()}
        
        if temperature == 0:  # 決定論的に選択
            best_action = max(visits.items(), key=lambda x: x[1])[0]
            action_probs = {action: 1.0 if action == best_action else 0.0 for action in visits}
        else:  # 温度付きソフトマックス
            visits_temp = {action: count ** (1.0 / temperature) for action, count in visits.items()}
            total = sum(visits_temp.values())
            action_probs = {action: count / total for action, count in visits_temp.items()}
            
        return action_probs

    def select_action(self, temperature=0.0):
        """
        最適な行動を選択する（訪問回数に基づく確率分布）
        
        Args:
            temperature: 温度パラメータ（探索のランダム性を制御）
            
        Returns:
            選択された行動
        """
        # シーケンス図の「最適手選択」部分に対応
        # MCTSから確率分布を取得
        action_probs = self.get_action_probabilities(temperature)
        
        if temperature == 0:  # 決定論的に選択
            best_action = max(action_probs.items(), key=lambda x: x[1])[0]
            return best_action
        else:  # 確率的に選択
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            return np.random.choice(actions, p=probs)
    
    def update_with_move(self, action):
        """
        指定した行動でゲーム状態と探索木を更新する
        
        Args:
            action: 実行する行動
        """
        # 指定された行動に対応する子ノードが存在するか確認
        if action in self.root.children:
            # 子ノードをルートにする（部分木を再利用）
            self.root = self.root.children[action]
            self.root.parent = None  # 親への参照を切る
        else:
            # 対応する子ノードがない場合は新しいルートノードを作成
            self.root = MCTSNode()
        
        # ゲーム状態を更新
        self.game_state.move(action)
        
        # ルートノードが展開されていない場合は展開
        if not self.root.expanded:
            self._expand_root_node()
    
    def add_exploration_noise(self):
        """ルートノードにディリクレノイズを追加（探索を促進）"""
        actions = list(self.root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        
        # ノイズを追加（exploration_fractionの割合で混合）
        for i, action in enumerate(actions):
            child = self.root.children[action]
            child.prior = (1 - self.exploration_fraction) * child.prior + self.exploration_fraction * noise[i]




class MCTSTrainer:
    """強化学習のトレーナー基底クラス"""
    #TODO trainer.py

    def __init__(self, parameter, remote_memory=None):
        self.parameter = parameter
        self.remote_memory = remote_memory
        self.train_count = 0

    def train(self):
        """モデルを訓練する"""
        raise NotImplementedError("サブクラスで実装する必要があります")
