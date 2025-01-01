import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import kron

# Define constants
N = 7  # Number of spins in the chain
J = 1  # Coupling strength
B = 100  # Magnetic field strength
dt = 0.15  # Time step
hbar = 1  # Reduced Planck's constant (setting to 1 for simplicity)

# Define Hamiltonian components
def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def pauli_y():
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def identity():
    return np.eye(2, dtype=np.complex128)


def create_hamiltonian(J, B_fields, noise_level=0.1):
    N = len(B_fields)
    H = np.zeros((2 ** N, 2 ** N), dtype=np.complex128)

    # Adding Gaussian noise to B_fields
    noisy_B_fields = B_fields + noise_level * np.random.randn(N)

    # XY coupling terms
    for n in range(N - 1):
        H_xx = np.eye(1, dtype=np.complex128)
        H_yy = np.eye(1, dtype=np.complex128)
        for k in range(N):
            if k == n or k == n + 1:
                H_xx = kron(H_xx, pauli_x())
                H_yy = kron(H_yy, pauli_y())
            else:
                H_xx = kron(H_xx, identity())
                H_yy = kron(H_yy, identity())
        H += (J / 2) * (H_xx + H_yy)

    # External magnetic field terms
    for n in range(N):
        H_z = np.eye(1, dtype=np.complex128)
        for k in range(N):
            if k == n:
                H_z = kron(H_z, pauli_z())
            else:
                H_z = kron(H_z, identity())
        H += noisy_B_fields[n] * H_z

    return H


# Time evolution using the Schrödinger equation
def time_evolution(psi, H, dt):
    U = scipy.linalg.expm(-1j * H * dt / hbar)  # Use scipy's expm function
    return U @ psi

# Convert complex state to real representation
def complex_to_real(psi):
    return np.hstack((psi.real, psi.imag))

# Calculate fidelity between two quantum states
def fidelity(psi1, psi2):
    return np.abs(np.dot(np.conj(psi1), psi2))**2

# Deep Q-Learning setup
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size * 2  # Real and imaginary parts
        self.action_size = action_size
        self.memory = deque(maxlen=40000)
        self.gamma = 0.9  # Reward decay
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.2
        self.learning_rate = 0.01
        self.batch_size = 50
        self.train_start = 900
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.losses = []
        self.rewards = []

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for state, action, reward, next_state in minibatch:
            target = self.model.predict(state)
            if next_state is not None:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            else:
                target[0][action] = reward
            history = self.model.fit(state, target, epochs=1, verbose=0)
            self.losses.append(history.history['loss'][0])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Proximal Policy Optimization setup
class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size * 2  # Real and imaginary parts
        self.action_size = action_size
        self.gamma = 0.9  # Reward decay
        self.learning_rate = 0.01
        self.epsilon = 0.2  # Clipping parameter
        self.value_coefficient = 0.5
        self.entropy_coefficient = 0.01
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        self.memory = []
        self.rewards = []

    def _build_model(self):
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        dense1 = tf.keras.layers.Dense(120, activation='relu')(state_input)
        dense2 = tf.keras.layers.Dense(120, activation='relu')(dense1)
        output_action = tf.keras.layers.Dense(self.action_size, activation='softmax')(dense2)
        output_value = tf.keras.layers.Dense(1, activation='linear')(dense2)
        model = tf.keras.Model(inputs=state_input, outputs=[output_action, output_value])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        probs, _ = self.model(state)
        action = np.random.choice(self.action_size, p=np.squeeze(probs))
        return action

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.epsilon * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def replay(self):
        if not self.memory:
            return

        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        _, values = self.model.predict(states)
        _, next_values = self.model.predict(next_states)

        advantages = self.compute_advantages(rewards, values, next_values, dones)
        advantages = np.array(advantages)

        actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=self.action_size)
        with tf.GradientTape() as tape:
            probs, values = self.model(states)
            action_probs = tf.reduce_sum(probs * actions_one_hot, axis=1)
            old_probs = tf.stop_gradient(action_probs)
            ratios = action_probs / (old_probs + 1e-10)
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            critic_loss = tf.reduce_mean(tf.square(rewards + self.gamma * next_values * (1 - dones) - values))
            entropy_loss = -tf.reduce_mean(probs * tf.math.log(probs + 1e-10))
            loss = actor_loss + self.value_coefficient * critic_loss - self.entropy_coefficient * entropy_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.memory = []


# Ensemble Agent
class EnsembleAgent:
    def __init__(self, state_size, action_size):
        self.dqn_agent = DQNAgent(state_size, action_size)
        self.ppo_agent = PPOAgent(state_size, action_size)
        self.total_rewards_dqn = 0
        self.total_rewards_ppo = 0
        self.total_episodes_dqn = 0
        self.total_episodes_ppo = 0

    def act(self, state):
        action_dqn = self.dqn_agent.act(state)
        action_ppo = self.ppo_agent.act(state)

        weight_dqn = self.calculate_weight(self.total_rewards_dqn, self.total_episodes_dqn)
        weight_ppo = self.calculate_weight(self.total_rewards_ppo, self.total_episodes_ppo)

        total_weight = weight_dqn + weight_ppo
        weight_dqn /= total_weight
        weight_ppo /= total_weight

        if weight_ppo < weight_dqn:
            action = action_dqn
        else:
            action = action_ppo

        return action

    def calculate_weight(self, total_rewards, total_episodes):
        return (1 + total_rewards) / (1 + total_episodes)

    def remember(self, state, action, reward, next_state, done):
        self.dqn_agent.remember(state, action, reward, next_state)
        self.ppo_agent.remember(state, action, reward, next_state, done)

        if action == self.dqn_agent.act(state):
            self.total_rewards_dqn += reward
            self.total_episodes_dqn += 1
        else:
            self.total_rewards_ppo += reward
            self.total_episodes_ppo += 1

    def replay(self):
        self.dqn_agent.replay()
        self.ppo_agent.replay()

# Hyper-parameters
state_size = 2 ** N  # Quantum state size
action_size = 2  # Two possible actions: control on/off
ensemble_agent = EnsembleAgent(state_size, action_size)
C = 200  # Update target network period
M = 1000  # Total episodes for training

# Storage for plotting
episode_rewards_ensemble = []
hundred_episode_rewards = []
episode_losses = []

# Define target state
target_psi = np.random.rand(state_size) + 1j * np.random.rand(state_size)
target_psi /= np.linalg.norm(target_psi)

# Simulating the environment and training the agents
termination_fidelity_threshold = 0.99
num_runs = 5
lowest_episode_reached = float('inf')
highest_fidelity_reached = 0.0
highest_fidelity_terminated = 0.0

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")
    ensemble_agent = EnsembleAgent(state_size, action_size)  # Reset the agent for each run
    episode_rewards_ensemble = []
    hundred_episode_rewards = []
    highest_fidelity_this_run = 0.0

    for episode in range(M):
        psi = np.random.rand(state_size) + 1j * np.random.rand(state_size)
        psi /= np.linalg.norm(psi)
        B_fields = np.zeros(N)
        total_reward = 0
        episode_terminated_early = False  # Flag to indicate if the episode should terminate

        for t in range(0, int(1 / dt)):
            state = complex_to_real(psi).reshape(1, -1)

            # Ensemble Agent's action
            action = ensemble_agent.act(state)

            if action == 1:
                B_fields[:] = B  # Apply the magnetic field B to all spins
            else:
                B_fields[:] = 0  # Set the magnetic field to 0 for all spins

            H = create_hamiltonian(J, B_fields)
            next_psi = time_evolution(psi, H, dt)
            next_psi /= np.linalg.norm(next_psi)  # Normalize the state

            # Modify the reward calculation
            reward = fidelity(next_psi, target_psi)  # Using fidelity as reward
            distance_reward = 1 - np.linalg.norm(next_psi - target_psi)  # Distance-based reward
            reward += distance_reward

            total_reward += reward
            next_state = complex_to_real(next_psi).reshape(1, -1)
            done = (t == int(1 / dt) - 1)

            # Store experience in Ensemble agent
            ensemble_agent.remember(state, action, reward, next_state, done)

            psi = next_psi

            # Train Ensemble agent
            ensemble_agent.replay()

            # Check if the fidelity threshold is reached
            current_fidelity = fidelity(next_psi, target_psi)
            if current_fidelity >= termination_fidelity_threshold:
                print(f"Episode {episode} terminated early at timestep {t} with fidelity {current_fidelity:.4f}")
                episode_terminated_early = True
                if episode < lowest_episode_reached:
                    lowest_episode_reached = episode
                if current_fidelity > highest_fidelity_terminated:
                    highest_fidelity_terminated = current_fidelity
                break

            # Track the highest fidelity reached
            if current_fidelity > highest_fidelity_this_run:
                highest_fidelity_this_run = current_fidelity

            # If the state does not change significantly, end the episode
            if np.linalg.norm(next_psi - psi) < 1e-5:
                break

        episode_rewards_ensemble.append(total_reward)

        if episode_terminated_early:
            break

        # Store total rewards every 100 episodes
        if (episode + 1) % 100 == 0:
            hundred_episode_reward = np.sum(episode_rewards_ensemble[-100:])
            hundred_episode_rewards.append(hundred_episode_reward)
            print(f"Training Episodes {episode - 99}-{episode} - Total Reward: {hundred_episode_reward:.4f}")

    # Update the highest fidelity reached across all runs if threshold not met
    if highest_fidelity_this_run > highest_fidelity_reached:
        highest_fidelity_reached = highest_fidelity_this_run

# Display results
if lowest_episode_reached == float('inf'):
    print(f"Termination fidelity threshold not reached. Highest fidelity achieved: {highest_fidelity_reached:.4f}")
else:
    print(f"Lowest episode where termination fidelity threshold was reached: {lowest_episode_reached}")
    print(f"Highest fidelity among the terminated episodes: {highest_fidelity_terminated:.4f}")

# Plotting results
plt.figure(figsize=(8, 6))

# Plot Training Episode vs Total Reward (every 100 episodes)
plt.subplot(1, 2, 1)
plt.plot(range(100, M + 1, 100), hundred_episode_rewards, label='Ensemble Agent')
plt.xlabel('Episode')
plt.ylabel('Total Reward (per 100 episodes)')
plt.title('Training Episode vs Total Reward (per 100 episodes)')
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# Calculate and print average testing rewards
average_reward = np.mean(episode_rewards_ensemble)
print("Average Reward:", average_reward)
print("Rewards:", hundred_episode_rewards)
print("Training complete.")
