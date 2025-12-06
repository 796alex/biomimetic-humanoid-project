"""
Baby Learning Algorithm - Neuromorphic brain that learns like a human infant
"""

import numpy as np
import tensorflow as tf
from collections import deque
import pickle
import time

class BabyBrain:
    def __init__(self, num_joints=20, num_muscles=40):
        self.num_joints = num_joints
        self.num_muscles = num_muscles
        
        # Body schema (initially empty - learns through experience)
        self.body_schema = np.zeros((num_muscles, num_joints))
        self.motor_map = {}
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Build neural networks
        self.build_neural_networks()
        
        # Learning parameters
        self.learning_rate = 0.001
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.exploration_min = 0.01
        
        print(f"BabyBrain initialized: {num_joints} joints, {num_muscles} muscles")
    
    def build_neural_networks(self):
        """Build lightweight neural networks for edge deployment"""
        # Actor network (policy)
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.num_joints*3,)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.num_muscles, activation='tanh')
        ])
        
        # Critic network (value)
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', 
                                 input_shape=(self.num_joints*3 + self.num_muscles,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate*2)
    
    def motor_babbling_phase(self, num_episodes=1000):
        """
        Phase 1: Motor babbling - random exploration to discover body capabilities
        Similar to how human infants learn through random movements
        """
        print("=== PHASE 1: MOTOR BABBLING ===")
        
        for episode in range(num_episodes):
            # Generate random motor commands
            if np.random.random() < self.exploration_rate:
                actions = np.random.uniform(-1, 1, self.num_muscles)
            else:
                # Use learned policy
                state = self.get_current_state()
                actions = self.actor.predict(state.reshape(1, -1))[0]
            
            # Execute actions (in simulation or hardware)
            sensory_feedback = self.execute_actions(actions)
            
            # Learn body schema through Hebbian learning
            self.update_body_schema(actions, sensory_feedback)
            
            # Calculate intrinsic reward (curiosity-driven)
            reward = self.calculate_intrinsic_reward(sensory_feedback)
            
            # Store experience
            self.experience_buffer.append({
                'state': self.get_current_state(),
                'action': actions,
                'reward': reward,
                'next_state': sensory_feedback,
                'done': False
            })
            
            # Train from experience
            if len(self.experience_buffer) > 32:
                self.train_from_experience(batch_size=32)
            
            # Decay exploration rate
            self.exploration_rate = max(self.exploration_min, 
                                       self.exploration_rate * self.exploration_decay)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Exploration rate = {self.exploration_rate:.3f}, "
                      f"Reward = {reward:.3f}")
    
    def update_body_schema(self, actions, sensory_feedback):
        """Hebbian learning: Neurons that fire together, wire together"""
        for muscle_idx, activation in enumerate(actions):
            if abs(activation) > 0.3:  Significant activation
                for joint_idx, joint_value in enumerate(sensory_feedback[:self.num_joints]):
                    if abs(joint_value) > 0.1:  Significant sensory change
                        # Hebbian update
                        self.body_schema[muscle_idx, joint_idx] += 0.01 * activation * joint_value
        
        # Normalize to prevent runaway growth
        self.body_schema = np.clip(self.body_schema, -1, 1)
    
    def calculate_intrinsic_reward(self, sensory_feedback):
        """
        Intrinsic motivation: Reward novelty and learning progress
        Similar to human curiosity
        """
        # 1. Novelty reward (explore new states)
        novelty = np.std(sensory_feedback)  # Higher variance = more novel
        
        # 2. Learning progress reward (reduce prediction error)
        predicted_state = self.predict_next_state(sensory_feedback)
        prediction_error = np.mean(np.abs(sensory_feedback - predicted_state))
        learning_reward = 1.0 / (1.0 + prediction_error)
        
        # 3. Control reward (achieve desired states)
        # (Implement based on specific goals)
        
        total_reward = novelty * 0.7 + learning_reward * 0.3
        return total_reward
    
    def train_from_experience(self, batch_size=32):
        """Deep Reinforcement Learning from stored experiences"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare training data
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        
        # Update critic (value function)
        with tf.GradientTape() as tape:
            # Current Q-values
            current_q = self.critic(tf.concat([states, actions], axis=1))
            
            # Target Q-values
            next_actions = self.actor(next_states)
            next_q = self.critic(tf.concat([next_states, next_actions], axis=1))
            target_q = rewards + 0.99 * next_q.numpy()  # Discount factor
            
            # Critic loss
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Update actor (policy)
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            new_q = self.critic(tf.concat([states, new_actions], axis=1))
            actor_loss = -tf.reduce_mean(new_q)  # Maximize Q-value
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
    
    def get_current_state(self):
        """Get current proprioceptive state"""
        # This would interface with real sensors
        # For simulation, return synthetic data
        return np.random.uniform(-1, 1, self.num_joints * 3)
    
    def execute_actions(self, actions):
        """Execute motor commands and return sensory feedback"""
        # Interface with hardware or simulation
        # For now, return simulated feedback
        feedback = np.zeros(self.num_joints * 3)
        
        # Simulate effect of actions on joints
        for i in range(self.num_joints):
            # Simple simulation: actions affect corresponding joints
            if i < len(actions):
                feedback[i] = actions[i] * 0.5 + np.random.normal(0, 0.1)
                feedback[i + self.num_joints] = actions[i] * 0.3  # Velocity
                feedback[i + self.num_joints*2] = actions[i] * 0.1  # Acceleration
        
        return feedback
    
    def predict_next_state(self, current_state):
        """Predict next sensory state given current state"""
        return self.actor.predict(current_state.reshape(1, -1))[0]
    
    def save_progress(self, filename="baby_brain_progress.pkl"):
        """Save learning progress"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'body_schema': self.body_schema,
                'actor_weights': self.actor.get_weights(),
                'critic_weights': self.critic.get_weights(),
                'exploration_rate': self.exploration_rate
            }, f)
        print(f"Progress saved to {filename}")
    
    def load_progress(self, filename="baby_brain_progress.pkl"):
        """Load learning progress"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.body_schema = data['body_schema']
            self.actor.set_weights(data['actor_weights'])
            self.critic.set_weights(data['critic_weights'])
            self.exploration_rate = data['exploration_rate']
        print(f"Progress loaded from {filename}")

# Example usage
if __name__ == "__main__":
    print("Initializing Baby Brain...")
    brain = BabyBrain(num_joints=20, num_muscles=40)
    
    print("\nStarting motor babbling phase (100 episodes)...")
    brain.motor_babbling_phase(num_episodes=100)
    
    print("\nSaving progress...")
    brain.save_progress()
    
    print("\nBaby Brain training completed!")
    print(f"Body schema shape: {brain.body_schema.shape}")
    print(f"Experiences collected: {len(brain.experience_buffer)}")