# Quantum Control with Ensemble Reinforcement Learning

## Authors

- Farshad Rahimi Ghashghaei
- Nebrase Elmrabit
- Ayyaz-Ul-Haq Qureshi
- Adnan Akhunzada
- Mehdi Yousefi
- 
## Affiliations

- Farshad Rahimi Ghashghaei:  School of Computing and Digital Technology, Birmingham City University, Birmingham B4 7XG, UK
- Nebrase Elmrabit and Ayyaz-Ul-Haq Qureshi:  Department of Cyber Security and Networks, Glasgow Caledonian University, Glasgow G4 0BA, UK
- Adnan Akhunzada: College of Computing and Information Technology, University of Doha for Science and Technology, Doha,
 24449, Qatar.
- Mehdi Yousefi:  Independent Researcher, 12 Riverview Place, Glasgow G5 8EH, UK
- 
## Overview

The project includes three main approaches:

1. **Ensemble Reinforcement Learning**
2. **Gradient Ascent Pulse Engineering (GRAPE)**
3. **Model Predictive Control (MPC)**

## Ensemble Reinforcement Learning (ensemble_learning.py)

This script implements an ensemble reinforcement learning algorithm to optimize magnetic fields in a spin chain for achieving a desired quantum state and maximaize the fidelity.

**Key Components:**

- **Hamiltonian Construction:** Uses Pauli matrices to create the Hamiltonian of the spin chain.
- **Time Evolution:** Simulates the time evolution of the quantum state.
- **Policy and Value Networks:** Neural networks to represent the policy and value functions.
- **Training Loop:** Iteratively updates the policy based on rewards and values to maximize fidelity.

## Conclusion

The ensemble agent demonstrates superior performance compared to standard methods, including Deep Q-Network (DQN) and Proximal Policy Optimization (PPO), by effectively using multiple learning strategies. It also surpasses both GRAPE and robust MPC approaches, even in challenging noisy conditions. This underscores the ensemble agent's advanced capability in optimizing quantum control tasks, offering a more reliable and efficient solution for achieving desired quantum states. Its robustness and adaptability make it a compelling choice for complex quantum control scenarios.
