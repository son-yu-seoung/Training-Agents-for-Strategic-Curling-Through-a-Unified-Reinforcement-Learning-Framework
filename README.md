# Training Agents for Strategic Curling Through a Unified Reinforcement Learning Framework

### Abstract 
Curling presents a challenging continuous-control problem in which shot outcomes depend on long-horizon interactions between complex physical dynamics, strategic intent, and opponent responses. Despite recent progress in applying reinforcement learning (RL) to games and sports, curling lacks a unified environment that jointly supports stable, rule-consistent simulation, structured state abstraction, and scalable agent training. To address this gap, we introduce a comprehensive learning framework for curling AI, consisting of a full-sized simulation environment, a task-aligned Markov decision process (MDP) formulation, and a two-phase training strategy designed for stable long-horizon optimization. First, we propose a novel MDP formulation that incorporates stone configuration, game context, and dynamic scoring factors, enabling an RL agent to reason simultaneously about physical feasibility and strategic desirability. Second, we present a two-phase curriculum learning procedure that significantly improves sample efficiency: Phase 1 trains the agent to master delivery mechanics by rewarding accurate placement around the tee line, while Phase 2 transitions to strategic learning with score-based rewards that encourage offensive and defensive planning. This staged training stabilizes policy learning and reduces the difficulty of direct exploration in the full curling action space. We integrate this MDP and training procedure into a unified Curling RL Framework, built upon a custom simulator designed for stability, reproducibility, and efficient RL training and a self-play mechanism tailored for strategic decision-making. Agent policies are optimized using Soft Actor-Critic (SAC), an entropy regularized off-policy algorithm designed for continuous control. As a case study, we compare the learned agent’s shot patterns with elite match records from the men’s division of the Le Gruyère AOP European Curling Championships 2023, using 6,512 extracted shot images. Experimental results demonstrate that the proposed framework learns diverse, human-like curling shots and outperforms ablated variants across both learning curves and head-to-head evaluations. Beyond curling, our framework provides a principled template for developing RL agents in physics-driven, strategy-intensive sports environments.

![GUI 이미지](curling_sheet.jpg)


---
# Getting Started
- Clone this repo:

```
git clone https://github.com/son-yu-seoung/Training-Agents-for-Strategic-Curling-Through-a-Unified-Reinforcement-Learning-Framework.git
cd Training-Agents-for-Strategic-Curling-Through-a-Unified-Reinforcement-Learning-Framework
```

- Install ours requirments:

```
pip install -r requirements.txt 
```

- Train
```
python train_phase1.py 
python train_phase2.py 
```

- Test

```
python evaluation_model.py  
```
