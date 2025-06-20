"""
自己対戦による強化学習データ生成
"""

import numpy as np
import time
import logging

# ロギング設定
logger = logging.getLogger(__name__)

class SelfPlay:
    """
    自己対戦クラス
    
    モデルを使用して自己対戦を行い、トレーニングデータを生成します。
    """
    
    def __init__(self, game_env, actor_critic, mcts_simulations=400, temperature=1.0):
        """
        自己対戦の初期化
        
        Args:
            game_env: ゲーム環境
            actor_critic: ActorCriticネットワーク
            mcts_simulations: MCTSのシミュレーション回数
            temperature: 行動選択の温度パラメータ
        """
        self.game_env = game_env
        self.actor_critic = actor_critic
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
    
    def play_game(self):
        """
        1ゲームの自己対戦を実行する
        
        Returns:
            (状態のリスト, 行動のリスト, 報酬のリスト, 最終結果)
        """
        from src.rl.mcts import MCTS
        
        # ゲームの初期化
        game = self.game_env.clone()
        game.reset()
        
        # MCTSの初期化
        mcts = MCTS(game, self.actor_critic)
        
        # 対局データの保存用
        states = []
        actions = []
        rewards = []
        
        # 最大手数（将棋は通常300手程度）
        max_moves = 500
        
        for move_num in range(max_moves):
            # 現在の状態を保存
            state_features = game.get_features()
            states.append(state_features)
            
            # MCTSで行動を選択
            for _ in range(self.mcts_simulations):
                mcts.search()
                
            # 温度パラメータを調整（序盤は探索的、終盤は最適）
            if move_num < 30:
                temp = self.temperature
            else:
                temp = 0.1
                
            # 行動確率の取得
            action, action_probs = mcts.get_action_probabilities(temperature=temp)
            
            # 行動を実行
            game.move(action)
            actions.append(action)
            
            # 報酬（勝敗）を確認
            if game.is_terminal():
                result = game.get_result(game.current_player)
                rewards.extend([result] * len(states))
                break
            
            # 報酬は終了時以外は0
            rewards.append(0.0)
            
            # 探索木を更新
            mcts.update_with_move(action)
            
        return states, actions, rewards, game.get_result(0)  # 先手視点の結果
    
    def generate_training_data(self, num_games=10):
        """
        複数ゲームの自己対戦を実行し、トレーニングデータを生成する
        
        Args:
            num_games: 対局数
            
        Returns:
            (状態のバッチ, 行動のバッチ, 報酬のバッチ)
        """
        all_states = []
        all_actions = []
        all_rewards = []
        results = []
        
        for i in range(num_games):
            start_time = time.time()
            states, actions, rewards, result = self.play_game()
            
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            results.append(result)
            
            end_time = time.time()
            logger.info(f"ゲーム {i+1}/{num_games} 完了 - 所要時間: {end_time - start_time:.1f}秒, 結果: {result}")
        
        # 統計情報
        win_rate = results.count(1.0) / len(results)
        draw_rate = results.count(0.0) / len(results)
        loss_rate = results.count(-1.0) / len(results)
        
        logger.info(f"自己対戦結果 - 勝率: {win_rate:.2f}, 引分率: {draw_rate:.2f}, 敗率: {loss_rate:.2f}")
        
        return np.array(all_states), np.array(all_actions), np.array(all_rewards) 