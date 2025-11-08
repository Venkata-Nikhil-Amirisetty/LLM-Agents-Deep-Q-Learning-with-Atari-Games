# LLM-Agents-Deep-Q-Learning-with-Atari-Games

# Deep Q-Learning for Atari Centipede

Implementation of Deep Q-Network (DQN) agent that learns to play Atari's Centipede game through reinforcement learning.

## Overview

This project implements a Deep Q-Learning agent with experience replay and target networks to master the classic Atari game Centipede. The agent learns optimal gameplay strategies through trial-and-error interaction with the environment.

### Key Features
- **Complete DQN implementation** with convolutional neural networks
- **Experience replay buffer** for stable learning
- **Multiple exploration strategies** including ε-greedy and Boltzmann
- **Comprehensive hyperparameter experiments** with detailed analysis
- **5000 training episodes** with systematic performance tracking

## Performance Results

### Baseline Configuration
- **Parameters:** α=0.7, γ=0.8, ε=1.0→0.01
- **Training:** 5000 episodes, max_steps=99
- **Results:** Converged to near-optimal play within environment constraints

### Experimental Improvements
- **Alternative α/γ values:** Tested conservative (α=0.5, γ=0.9) vs aggressive (α=0.9, γ=0.7) learning
- **Boltzmann exploration:** Implemented temperature-based action selection with max_steps=500
- **Decay rate variations:** Compared fast (0.02), slow (0.005), and baseline (0.01) epsilon decay

## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for faster training)
- 8GB+ RAM

### Installation

1. Clone or download this repository
2. Install required packages:
```bash
pip install gymnasium[atari,accept-rom-license]
pip install torch torchvision
pip install numpy matplotlib
```
3. Open Jupyter Notebook:
```bash
jupyter notebook
```
4. Open `dql_centipede_notebook.ipynb`
5. Run all cells sequentially

### Quick Start
- **For testing only:** Skip to Cell 9 and load pre-trained weights
- **For full training:** Run all cells (2-3 hours on GPU)
- **For experiments:** Run Cells 11-13 for hyperparameter comparisons

## Technical Implementation

### Architecture
- **Input:** Preprocessed frames (1×105×80 grayscale)
- **CNN Layers:** 32, 64, 64 filters with ReLU activation
- **FC Layers:** 512 hidden units → 18 action outputs
- **Optimizer:** Adam with lr=0.0001

### Key Components
1. **Deep Q-Network:** Approximates Q-values for continuous state space
2. **Experience Replay:** 10,000 capacity buffer for stable learning
3. **Target Network:** Updated every 100 steps for training stability
4. **Exploration Strategies:** ε-greedy and Boltzmann (softmax) policies 

## Code Attribution

**Original Implementation:**
- Complete DQN architecture and training loop
- Experience replay buffer system
- Visualization and analysis functions
- Boltzmann exploration enhancement
- All hyperparameter experiments

**Frameworks Used:**
- **PyTorch:** Neural network framework (Apache 2.0 License)
- **Gymnasium:** Atari environment (MIT License)
- **NumPy/Matplotlib:** Data processing and visualization (BSD License)

**Conceptual References:**
- Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- Sutton & Barto - Reinforcement Learning textbook
- Standard Atari preprocessing techniques

## License

MIT License - Free for academic and commercial use with attribution.


## Acknowledgments

- OpenAI Gym/Gymnasium community for environment support
- PyTorch team for excellent documentation
- Course instructors for assignment design
