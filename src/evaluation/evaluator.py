def __init__(self, model_path="data/trained_params.msgpack", config=None):
    """評価器の初期化
    
    Args:
        model_path: モデルパラメータのパス
        config: 設定
    """
    # モデルの初期化
    self.model = SwinShogiModel()
    
    # パラメータのロード
    try:
        self.params = load_params(model_path)
        print(f"モデルパラメータをロードしました: {model_path}")
    except Exception as e:
        print(f"モデルパラメータのロードに失敗しました: {e}")
        # 初期化
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 9, 9, 44))
        self.params = self.model.init(key, dummy_input)
        print("ランダム初期化したパラメータを使用します")