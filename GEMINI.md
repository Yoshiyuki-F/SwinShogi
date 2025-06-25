Answer me in Japanese.


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

This project requires Python 3.13 and uses Poetry for dependency management:

```bash
pyenv local 3.13
poetry install
```

## Common Commands

### Testing
- **Run all tests**: `./scripts/test_all.sh`
- **Run integration tests**: `./scripts/run_integration_test.sh` 
- **Run specific test module**: `PYTHONPATH="$(pwd)" python -m unittest tests.rl.test_mcts`
- **Run legacy internal tests**: `./scripts/run_tests.sh`

### Training
- **Start training**: `poetry run train` (configured in pyproject.toml)

### Environment Variables
- Set `JAX_LOG_LEVEL=WARNING` to reduce JAX verbosity during testing
- `PYTHONPATH` must include project root for module imports

## Core Architecture Flow

SwinShogi implements a reinforcement learning system for Japanese chess with this key pipeline:

1. **Board State Encoding** (`src/shogi/board_encoder.py`)
   - Converts 9x9 shogi board to 2-channel feature representation
   - Encodes piece positions and ownership in separate channels

2. **Swin Transformer Feature Extraction** (`src/model/swin_transformer.py` → `src/model/shogi_model.py`)
   - Processes board features through windowed attention mechanism
   - Uses 1x1 patches (each square is a patch) with window_size=3
   - 2-layer architecture: 9x9 → 3x3 with patch merging factor of 3

3. **Actor-Critic Prediction** (`src/model/actor_critic.py`)
   - Wraps SwinShogiModel to output policy probabilities and state values
   - Policy head outputs 2187 logits (9×9×27 possible moves)
   - Value head outputs single scalar for position evaluation

4. **MCTS Search** (`src/rl/mcts.py`)
   - Uses Actor-Critic predictions to guide tree search
   - Expands nodes based on policy priors and UCB selection
   - Performs backpropagation to update node values

5. **Training Loop** (`src/rl/trainer.py`)
   - Implements policy gradient methods with cross-entropy loss
   - Uses self-play data generation (`src/rl/self_play.py`)
   - Gradient clipping and JAX JIT optimization

## Key Configuration

Model hyperparameters are centralized in `config/default_config.py`:
- Embed dimension: 96
- Window size: 3 (matches shogi piece movement patterns)
- Policy outputs: 2187 (9×9×27 for all possible moves)
- Patch merge factor: 3 (shogi-specific, differs from standard Swin Transformer's factor of 2)

MCTS hyperparameters:
- Simulations: 400 (configurable)
- Max depth: 200 (prevents infinite search)
- C_PUCT: Dynamic calculation with init=1.25, base=19652
- Exploration noise: Dirichlet(alpha=0.3) with epsilon=0.25
- FPU reduction: 0.25 (First Play Urgency)

## Testing Strategy

The project uses both internal tests (`src/tests/`) and standard external tests (`tests/`):
- **Unit tests**: Individual component testing (MCTS, model components)
- **Integration tests**: End-to-end pipeline testing (Transformer → Actor-Critic → MCTS)
- **GPU detection**: Integration test script automatically detects and reports GPU availability

## JAX/Flax Patterns

- All models use Flax Linen modules
- JIT compilation for inference performance (`@jax.jit` decorators)
- Gradient processing utilities in `src/utils/model_utils.py`
- Device detection and setup in `src/utils/jax_utils.py`

## Shogi Domain Knowledge

- Complete rule implementation including illegal move detection
- SFEN notation support for position encoding/decoding
- USI protocol interface (`src/interface/usi.py`) for engine communication
- Board visualization utilities for debugging

## Important Implementation Details

- The Swin Transformer is adapted for 9x9 shogi boards vs standard computer vision applications
- Policy output dimension accounts for all legal shogi moves including piece drops
- MCTS uses domain-specific move generation from shogi rule engine
- Training uses self-play to generate position-value pairs without requiring labeled data

## MCTS Implementation

The MCTS (`src/rl/mcts.py`) provides:
- **Node Management**: MCTSNode class with PUCT selection and value backup
- **Search Tree**: Efficient tree traversal with depth limits and terminal detection
- **Neural Integration**: Seamless Actor-Critic prediction calls via `predict_for_mcts()`
- **Game State Management**: Automatic ShogiGame state cloning and move application
- **Exploration**: Dirichlet noise at root for training diversity
- **Utilities**: `create_mcts()` and `mcts_search()` convenience functions

Usage:
```python
# Simple search
action_probs, search_info = mcts_search(model, params, game_state, n_simulations=100)

# Advanced usage
mcts = create_mcts(model, params, max_depth=50, c_puct=1.5)
action_probs, root_node = mcts.search(game_state)
selected_action = mcts.select_action(action_probs, temperature=0.0)
```