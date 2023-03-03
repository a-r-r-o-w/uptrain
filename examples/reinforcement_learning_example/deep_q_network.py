import tensorflow as tf
import numpy as np

from typing import List


class DeepQNetwork(tf.keras.Model):
    def __init__(
        self, num_actions: int, input_dims: List[int], hidden_layer_dims: List[int]
    ):
        super(DeepQNetwork, self).__init__()

        self.num_actions = num_actions
        self.input_dims = input_dims
        self.hidden_layer_dims = hidden_layer_dims

        self.hidden_layers = []
        self.hidden_layers.append(tf.keras.layers.Flatten(input_shape=input_dims))
        self.hidden_layers.extend(
            [
                tf.keras.layers.Dense(units=units, activation=tf.nn.relu)
                for units in hidden_layer_dims
            ]
        )
        self.Q = tf.keras.layers.Dense(units=num_actions, activation=None)

    def call(self, state):
        x = state
        for layer in self.hidden_layers:
            x = layer(x)
        Q = self.Q(x)
        return Q

    def get_config(self):
        config = {}
        config["num_actions"] = self.num_actions
        config["input_dims"] = self.input_dims
        config["hidden_layer_dims"] = self.hidden_layer_dims
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MemoryBuffer:
    def __init__(self, memory_size: int, input_shape: List[int]):
        self.memory_size = memory_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape), np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_shape), np.float32)
        self.action_memory = np.zeros(self.memory_size, np.int32)
        self.reward_memory = np.zeros(self.memory_size, np.float32)
        self.terminal_memory = np.zeros(self.memory_size, np.bool8)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.memory_counter += 1

    def sample_buffer(self, batch_size: int):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size, replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]
        return states, actions, rewards, new_states, terminals


class Agent:
    def __init__(
        self,
        num_actions: int,
        input_dims: List[int],
        *,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        epsilon: float = 1,
        epsilon_decrement: float = 0.001,
        min_epsilon: float = 0.01,
        memory_size: int = 100000,
        filename: str = "dqn_agent",
        hidden_dims: List[int] = [128, 128],
        memory_replace_after=100
    ):
        self.num_actions = num_actions
        self.input_dims = input_dims
        self.action_space = list(range(num_actions))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decrement = epsilon_decrement
        self.min_epsilon = min_epsilon
        self.memory_size = memory_size
        self.filename = filename
        self.hidden_dims = hidden_dims
        self.memory_replace_after = memory_replace_after
        self.learn_step_counter = 0
        self.memory = MemoryBuffer(memory_size, input_dims)
        self.q_main_network = DeepQNetwork(num_actions, input_dims, hidden_dims)
        self.q_target_network = DeepQNetwork(num_actions, input_dims, hidden_dims)
        self.q_main_network.compile(
            optimizer=tf.optimizers.Adam(learning_rate),
            loss=tf.losses.MeanSquaredError(),
        )
        self.q_target_network.compile(
            optimizer=tf.optimizers.Adam(learning_rate),
            loss=tf.losses.MeanSquaredError(),
        )

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = tf.convert_to_tensor([observation])
            actions = self.q_main_network.call(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action

    def predict_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions = self.q_main_network.call(state)
        action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        if self.learn_step_counter % self.memory_replace_after == 0:
            self.q_target_network.set_weights(self.q_main_network.get_weights())

        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size
        )
        q_pred = self.q_main_network(states)
        q_next = tf.math.reduce_max(
            self.q_target_network(states_), axis=1, keepdims=True
        ).numpy()
        q_target = np.copy(q_pred)

        for index, terminal in enumerate(dones):
            if terminal:
                q_next[index] = 0
            q_target[index, actions[index]] = (
                rewards[index] + self.gamma * q_next[index]
            )

        self.q_main_network.train_on_batch(states, q_target)
        self.epsilon = (
            self.epsilon - self.epsilon_decrement
            if self.epsilon > self.min_epsilon
            else self.min_epsilon
        )
        self.learn_step_counter += 1

    def save_model(self):
        self.q_main_network.save(self.filename)

    def load_model(self):
        self.q_main_network = tf.keras.models.load_model(
            self.filename, custom_objects={"DeepQNetwork": DeepQNetwork}
        )
