"""
USIプロトコルインターフェース
"""

import sys
import os
import time
import jax


# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ..model.swin_shogi import SwinShogiModel
from ..utils.model_utils import load_params
from ..shogi.board_encoder import encode_position
from ..shogi.shogi_pieces import *
from ..rl.rl_config import RLConfig
from ..rl.mcts import MCTS, MCTSConfig
from ..rl.shogi_env import ShogiEnv
from config import USI_CONFIG, PATHS
import subprocess

import threading
import logging
import numpy as np
from src.model.swin_shogi import create_swin_shogi_model
from src.shogi.shogi_pieces import ShogiGame
from src.rl.mcts import MCTS
from src.core.reinforcement_learning import ActorCritic

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("usi_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class USIInterface:
    """
    USIプロトコルを実装するインターフェースクラス
    
    このクラスは以下の機能を提供します：
    - USIコマンドの解析と応答
    - SFENフォーマットの解析と将棋盤の状態の管理
    - SwinShogiモデルを使用した思考処理
    - 外部将棋エンジンとの対局
    """
    
    def __init__(self):
        """USIインターフェースの初期化"""
        # エンジン情報の設定
        self.engine_name = "SwinShogi"
        self.engine_author = "SwinShogiチーム"
        self.engine_version = "0.1.0"
        
        # モデルとゲーム状態
        self.model = None
        self.params = None
        self.game = ShogiGame()
        self.mcts = None
        self.actor_critic = None
        
        # 設定パラメータ
        self.search_limits = {
            "time": 1000,  # ミリ秒単位の思考時間
            "nodes": 10000,  # 探索ノード数
            "depth": 10,    # 探索深さ
        }
        self.model_path = "data/models/swin_shogi_model.pkl"
        
        # 思考スレッド
        self.thinking_thread = None
        self.stop_thinking = False
        
        # 初期化状態
        self.initialized = False
        self.ready = False
        
        # コマンドマップの設定
        self.command_map = {
            "usi": self._cmd_usi,
            "isready": self._cmd_isready,
            "setoption": self._cmd_setoption,
            "usinewgame": self._cmd_usinewgame,
            "position": self._cmd_position,
            "go": self._cmd_go,
            "stop": self._cmd_stop,
            "quit": self._cmd_quit,
            "gameover": self._cmd_gameover,
        }
    
    def run(self):
        """
        USIインターフェースの実行ループ
        
        標準入力からコマンドを読み取り、対応するハンドラを呼び出します。
        """
        try:
            while True:
                cmd_line = input().strip()
                if not cmd_line:
                    continue
                    
                logger.debug(f"Received command: {cmd_line}")
                
                # コマンドの解析
                cmd_parts = cmd_line.split()
                cmd = cmd_parts[0]
                
                if cmd in self.command_map:
                    self.command_map[cmd](cmd_line, cmd_parts)
                    if cmd == "quit":
                        break
                else:
                    logger.warning(f"Unknown command: {cmd}")
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception(f"Error in USI interface: {e}")
            
        logger.info("USI interface terminated")
    
    def _cmd_usi(self, cmd_line: str, cmd_parts: List[str]):
        """
        'usi'コマンドの処理
        
        エンジン情報とオプションを出力します。
        """
        print(f"id name {self.engine_name} {self.engine_version}")
        print(f"id author {self.engine_author}")
        
        # オプションの設定
        print("option name ModelPath type string default data/models/swin_shogi_model.pkl")
        print("option name Threads type spin default 1 min 1 max 32")
        print("option name NodeLimit type spin default 10000 min 100 max 1000000")
        print("option name TimeLimit type spin default 1000 min 10 max 600000")
        
        print("usiok")
    
    def _cmd_isready(self, cmd_line: str, cmd_parts: List[str]):
        """
        'isready'コマンドの処理
        
        モデルをロードし、準備ができたことを通知します。
        """
        if not self.initialized:
            try:
                # モデルの初期化
                logger.info("Initializing model...")
                rng = jax.random.PRNGKey(0)
                self.model, self.params = create_swin_shogi_model(rng)
                
                try:
                    # モデルパラメータの読み込み
                    logger.info(f"Loading model from: {self.model_path}")
                    self.params = self.model.load_params(self.model_path)
                    logger.info("Model loaded successfully")
                except FileNotFoundError:
                    logger.warning(f"Model file not found: {self.model_path}")
                    logger.warning("Using initialized model")
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    logger.warning("Using initialized model")
                
                # MCTSとActorCriticの初期化
                self.actor_critic = ActorCritic(self.model, self.params)
                self.mcts = MCTS(self.game, self.actor_critic)
                
                self.initialized = True
                logger.info("Model initialization completed")
            except Exception as e:
                logger.exception(f"Error during initialization: {e}")
                print("readyok")  # エラーが発生しても準備完了と応答（USIプロトコル要件）
                return
        
        self.ready = True
        print("readyok")
    
    def _cmd_setoption(self, cmd_line: str, cmd_parts: List[str]):
        """
        'setoption'コマンドの処理
        
        エンジンのオプションを設定します。
        """
        if len(cmd_parts) < 4 or cmd_parts[1] != "name":
            logger.warning("Invalid setoption command")
            return
            
        option_name = cmd_parts[2]
        
        if "value" not in cmd_line:
            logger.warning(f"No value provided for option: {option_name}")
            return
            
        value_index = cmd_line.index("value") + 6
        value = cmd_line[value_index:].strip()
        
        if option_name == "ModelPath":
            self.model_path = value
            logger.info(f"Set model path to: {value}")
            
        elif option_name == "Threads":
            try:
                threads = int(value)
                if threads < 1:
                    threads = 1
                logger.info(f"Set threads to: {threads}")
                # 現在のところマルチスレッドサポートは未実装
            except ValueError:
                logger.warning(f"Invalid thread value: {value}")
                
        elif option_name == "NodeLimit":
            try:
                nodes = int(value)
                if nodes < 100:
                    nodes = 100
                self.search_limits["nodes"] = nodes
                logger.info(f"Set node limit to: {nodes}")
            except ValueError:
                logger.warning(f"Invalid node limit: {value}")
                
        elif option_name == "TimeLimit":
            try:
                time_ms = int(value)
                if time_ms < 10:
                    time_ms = 10
                self.search_limits["time"] = time_ms
                logger.info(f"Set time limit to: {time_ms}ms")
            except ValueError:
                logger.warning(f"Invalid time limit: {value}")
                
        else:
            logger.warning(f"Unknown option: {option_name}")
    
    def _cmd_usinewgame(self, cmd_line: str, cmd_parts: List[str]):
        """
        'usinewgame'コマンドの処理
        
        新しいゲームの準備をします。
        """
        # 新しいゲームの初期化
        self.game = ShogiGame()
        if self.actor_critic is not None:
            self.mcts = MCTS(self.game, self.actor_critic)
        logger.info("New game initialized")
    
    def _cmd_position(self, cmd_line: str, cmd_parts: List[str]):
        """
        'position'コマンドの処理
        
        盤面の状態を設定します。
        """
        if len(cmd_parts) < 2:
            logger.warning("Invalid position command")
            return
            
        if cmd_parts[1] == "startpos":
            # 初期局面
            self.game = ShogiGame()
            start_move_idx = 2
        elif cmd_parts[1] == "sfen":
            # SFEN形式の局面
            if len(cmd_parts) < 6:
                logger.warning("Invalid sfen format")
                return
                
            sfen = " ".join(cmd_parts[2:6])
            self.game.setup_custom_position(sfen)
            start_move_idx = 6
        else:
            logger.warning(f"Invalid position type: {cmd_parts[1]}")
            return
            
        # 指し手の適用
        if len(cmd_parts) > start_move_idx and cmd_parts[start_move_idx] == "moves":
            for move in cmd_parts[start_move_idx+1:]:
                success = self.game.move(move)
                if not success:
                    logger.warning(f"Failed to apply move: {move}")
                    break
                    
        logger.info("Position set up")
    
    def _cmd_go(self, cmd_line: str, cmd_parts: List[str]):
        """
        'go'コマンドの処理
        
        探索を開始します。
        """
        if not self.ready:
            logger.warning("Engine not ready")
            print("bestmove resign")
            return
            
        # 探索制限の解析
        self._parse_go_limits(cmd_line, cmd_parts)
        
        # 思考スレッドの作成と開始
        self.stop_thinking = False
        self.thinking_thread = threading.Thread(target=self._thinking_routine)
        self.thinking_thread.start()
    
    def _thinking_routine(self):
        """
        思考ルーチン
        
        MCTSを使用して最善手を探索します。
        """
        try:
            start_time = time.time()
            time_limit_ms = self.search_limits.get("time", 1000)
            node_limit = self.search_limits.get("nodes", 10000)
            
            # 探索の実行
            best_move = None
            visit_counts = None
            
            for iterations in range(node_limit):
                if self.stop_thinking:
                    break
                    
                elapsed = (time.time() - start_time) * 1000
                if elapsed > time_limit_ms:
                    break
                    
                self.mcts.search()
                
                # 定期的に進捗を報告
                if iterations % 100 == 0:
                    best_action, visit_counts = self.mcts.get_action_probabilities(temperature=0)
                    best_move = self.game.coords_to_move(*best_action)
                    logger.debug(f"Iteration {iterations}: best_move={best_move}")
                    
                    top_moves = sorted([(self.game.coords_to_move(*a), p) for a, p in visit_counts.items()], 
                                     key=lambda x: x[1], reverse=True)[:5]
                    
                    # 思考情報の出力（USIプロトコル形式）
                    principal_variation = " ".join([m for m, _ in top_moves[:3]])
                    print(f"info depth {iterations//100} time {int(elapsed)} nodes {iterations} "
                          f"score cp {int(self.mcts.root.value*100)} pv {principal_variation}")
            
            # 最終的な最善手の決定
            best_action, visit_counts = self.mcts.get_action_probabilities(temperature=0)
            best_move = self.game.coords_to_move(*best_action)
            
            # 結果を出力
            elapsed = (time.time() - start_time) * 1000
            top_moves = sorted([(self.game.coords_to_move(*a), p) for a, p in visit_counts.items()], 
                             key=lambda x: x[1], reverse=True)[:5]
            
            logger.info(f"Thinking completed: time={elapsed:.1f}ms, nodes={iterations}")
            logger.info(f"Best move: {best_move}")
            logger.info(f"Top moves: {top_moves}")
            
            print(f"bestmove {best_move}")
            
        except Exception as e:
            logger.exception(f"Error during thinking: {e}")
            print("bestmove resign")
    
    def _parse_go_limits(self, cmd_line: str, cmd_parts: List[str]):
        """
        'go'コマンドの探索制限を解析します
        """
        i = 1
        while i < len(cmd_parts):
            if cmd_parts[i] == "infinite":
                # 無制限探索（デフォルトのノード数制限を使用）
                self.search_limits["time"] = float('inf')
                i += 1
            elif cmd_parts[i] == "btime" and i + 1 < len(cmd_parts):
                # 先手の残り時間（ミリ秒）
                if self.game.current_player == 0:  # 先手番
                    remaining_time = int(cmd_parts[i+1])
                    self.search_limits["time"] = min(remaining_time // 30, 10000)  # 残り時間の1/30を使用、最大10秒
                i += 2
            elif cmd_parts[i] == "wtime" and i + 1 < len(cmd_parts):
                # 後手の残り時間（ミリ秒）
                if self.game.current_player == 1:  # 後手番
                    remaining_time = int(cmd_parts[i+1])
                    self.search_limits["time"] = min(remaining_time // 30, 10000)  # 残り時間の1/30を使用、最大10秒
                i += 2
            elif cmd_parts[i] == "byoyomi" and i + 1 < len(cmd_parts):
                # 秒読み時間（ミリ秒）
                byoyomi = int(cmd_parts[i+1])
                self.search_limits["time"] = min(byoyomi * 0.8, self.search_limits.get("time", float('inf')))
                i += 2
            elif cmd_parts[i] == "depth" and i + 1 < len(cmd_parts):
                # 探索深さ
                self.search_limits["depth"] = int(cmd_parts[i+1])
                i += 2
            elif cmd_parts[i] == "nodes" and i + 1 < len(cmd_parts):
                # 探索ノード数
                self.search_limits["nodes"] = int(cmd_parts[i+1])
                i += 2
            else:
                # その他のパラメータはスキップ
                i += 1
                
        logger.info(f"Search limits set to: {self.search_limits}")
    
    def _cmd_stop(self, cmd_line: str, cmd_parts: List[str]):
        """
        'stop'コマンドの処理
        
        探索を停止します。
        """
        self.stop_thinking = True
        if self.thinking_thread and self.thinking_thread.is_alive():
            self.thinking_thread.join(1.0)  # 1秒待機
            logger.info("Thinking stopped")
    
    def _cmd_quit(self, cmd_line: str, cmd_parts: List[str]):
        """
        'quit'コマンドの処理
        
        エンジンを終了します。
        """
        logger.info("Quit command received")
        self._cmd_stop(cmd_line, cmd_parts)
    
    def _cmd_gameover(self, cmd_line: str, cmd_parts: List[str]):
        """
        'gameover'コマンドの処理
        
        ゲーム終了を処理します。
        """
        if len(cmd_parts) > 1:
            result = cmd_parts[1]
            logger.info(f"Game over: {result}")
        else:
            logger.info("Game over")

def main():
    """USIインターフェースのエントリーポイント"""
    usi = USIInterface()
    usi.run()

if __name__ == "__main__":
    main()