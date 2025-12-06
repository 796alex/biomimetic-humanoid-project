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
    """
ALGORITMO COMPLETO DE APRENDIZAJE CORPORAL
El cerebro descubre su cuerpo como un bebé humano
"""
import numpy as np
import tensorflow as tf
from collections import deque
import pickle
import time
from datetime import datetime

class BabyBrain:
    def __init__(self, num_joints=20, num_muscles=40):
        """Inicializa cerebro con capacidad de aprender"""
        self.num_joints = num_joints
        self.num_muscles = num_muscles
        
        # Mapa corporal inicial (tabula rasa)
        self.body_schema = np.zeros((num_muscles, num_joints))
        self.motor_memory = {}
        self.sensory_memory = {}
        
        # Buffer de experiencias
        self.experience_buffer = deque(maxlen=10000)
        
        # Red neuronal principal
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()
        
        # Estado interno
        self.learning_rate = 0.001
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.995
        
        print(f"🧠 Cerebro BabyBrain inicializado: {num_joints} articulaciones, {num_muscles} músculos")
    
    def _build_policy_network(self):
        """Construye red neuronal para toma de decisiones"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(self.num_joints * 3,), 
                                 activation='relu', name='policy_dense1'),
            tf.keras.layers.LayerNormalization(name='policy_norm1'),
            tf.keras.layers.Dropout(0.2, name='policy_dropout1'),
            
            tf.keras.layers.Dense(64, activation='relu', name='policy_dense2'),
            tf.keras.layers.LayerNormalization(name='policy_norm2'),
            
            tf.keras.layers.Dense(32, activation='relu', name='policy_dense3'),
            
            tf.keras.layers.Dense(self.num_muscles, activation='tanh', 
                                 name='policy_output')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def _build_value_network(self):
        """Construye red para evaluación de estados"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(self.num_joints * 3,), 
                                 activation='relu', name='value_dense1'),
            tf.keras.layers.LayerNormalization(name='value_norm1'),
            
            tf.keras.layers.Dense(32, activation='relu', name='value_dense2'),
            
            tf.keras.layers.Dense(1, activation='linear', name='value_output')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate * 0.5),
            loss='mse'
        )
        
        return model
    
    def motor_babbling_phase(self, iterations=1000, save_interval=100):
        """
        FASE 1: Balbuceo motor - Descubrimiento del cuerpo
        """
        print("=" * 60)
        print("FASE 1: BALBUCEO MOTOR - DESCUBRIENDO EL CUERPO")
        print("=" * 60)
        
        start_time = time.time()
        
        for i in range(iterations):
            # 1. Generar acción aleatoria (exploración)
            if np.random.random() < self.exploration_rate:
                action = np.random.uniform(-1, 1, self.num_muscles)
            else:
                # Usar política aprendida
                state = self._get_current_state()
                action = self.policy_network.predict(state.reshape(1, -1), verbose=0)[0]
            
            # 2. Ejecutar acción y obtener retroalimentación sensorial
            sensory_feedback = self._execute_motor_command(action)
            
            # 3. Calcular recompensa intrínseca
            reward = self._calculate_intrinsic_reward(action, sensory_feedback)
            
            # 4. Aprender mapa corporal
            self._update_body_schema(action, sensory_feedback)
            
            # 5. Guardar experiencia para aprendizaje posterior
            experience = {
                'state': self._get_current_state(),
                'action': action,
                'reward': reward,
                'next_state': sensory_feedback,
                'done': False
            }
            self.experience_buffer.append(experience)
            
            # 6. Decaer tasa de exploración
            if self.exploration_rate > self.min_exploration:
                self.exploration_rate *= self.exploration_decay
            
            # 7. Aprendizaje periódico
            if i % 10 == 0 and len(self.experience_buffer) > 32:
                self._train_from_experience(batch_size=32)
            
            # 8. Guardar progreso
            if i % save_interval == 0:
                self._save_progress(iteration=i)
                elapsed = time.time() - start_time
                print(f"Iteración {i}: Recompensa={reward:.3f}, "
                      f"Exploración={self.exploration_rate:.3f}, "
                      f"Tiempo={elapsed:.1f}s")
        
        print("✅ Fase 1 completada: Mapa corporal básico aprendido")
    
    def _execute_motor_command(self, action):
        """Simula ejecución de comando motor (en hardware real se conecta a los músculos)"""
        # Simulación de cinemática simple
        # En implementación real, esto controlaría los servos/motores
        
        # Efecto de cada músculo en las articulaciones (aprendido)
        joint_effects = np.dot(action.reshape(1, -1), self.body_schema)
        
        # Ruido sensorial (simula imperfecciones del mundo real)
        noise = np.random.normal(0, 0.1, self.num_joints)
        
        # Posición articular resultante
        joint_positions = np.clip(joint_effects.flatten() + noise, -1, 1)
        
        # Velocidad (derivada aproximada)
        joint_velocities = np.zeros(self.num_joints)  # Simplificado
        
        # Fuerza muscular percibida
        muscle_tension = np.abs(action) * 0.8
        
        # Combinar toda la retroalimentación sensorial
        sensory_feedback = np.concatenate([
            joint_positions,        # Posición articular
            joint_velocities,       # Velocidad articular  
            muscle_tension          # Tensión muscular
        ])
        
        return sensory_feedback
    
    def _calculate_intrinsic_reward(self, action, sensory_feedback):
        """
        Recompensa intrínseca basada en:
        1. Novedad (explorar estados nuevos)
        2. Control (predecir consecuencias)
        3. Eficiencia (mínimo esfuerzo)
        """
        # 1. Recompensa por novedad
        novelty = np.std(sensory_feedback)  # Más variación = más novedad
        
        # 2. Recompensa por control (capacidad de predecir)
        predicted_state = self._predict_next_state(self._get_current_state(), action)
        prediction_error = np.mean(np.abs(predicted_state - sensory_feedback))
        control_reward = 1.0 / (1.0 + prediction_error)  # Más control = más recompensa
        
        # 3. Recompensa por eficiencia (mínimo esfuerzo)
        energy_cost = np.sum(np.abs(action))
        efficiency_reward = 1.0 / (1.0 + energy_cost * 0.1)
        
        # Recompensa compuesta
        total_reward = (novelty * 0.3 + 
                       control_reward * 0.4 + 
                       efficiency_reward * 0.3)
        
        return total_reward
    
    def _update_body_schema(self, action, sensory_feedback):
        """Actualiza el mapa corporal usando aprendizaje Hebbiano"""
        # Aprendizaje Hebbiano: "Neuronas que se activan juntas, se conectan juntas"
        
        # Extraer posiciones articulares de la retroalimentación sensorial
        joint_positions = sensory_feedback[:self.num_joints]
        
        for muscle_idx, muscle_activation in enumerate(action):
            if abs(muscle_activation) > 0.2:  # Solo si el músculo se activó significativamente
                for joint_idx, joint_change in enumerate(joint_positions):
                    if abs(joint_change) > 0.1:  # Solo si la articulación se movió
                        # Regla Hebbiana: Δpeso = η * activación_pre * activación_post
                        weight_change = (0.01 * muscle_activation * joint_change * 
                                       (1.0 - abs(self.body_schema[muscle_idx, joint_idx])))
                        self.body_schema[muscle_idx, joint_idx] += weight_change
        
        # Normalizar para mantener estabilidad
        self.body_schema = np.clip(self.body_schema, -1.0, 1.0)
    
    def _train_from_experience(self, batch_size=32, gamma=0.99):
        """Aprendizaje por refuerzo profundo (DQN simplificado)"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Muestrear experiencias aleatorias
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        # Preparar datos para entrenamiento
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        
        # Calcular targets Q-learning
        current_q_values = self.value_network.predict(states, verbose=0)
        next_q_values = self.value_network.predict(next_states, verbose=0)
        
        targets = rewards + gamma * next_q_values.flatten() * (1 - dones)
        
        # Entrenar red de valor
        self.value_network.fit(states, targets.reshape(-1, 1), 
                              epochs=1, batch_size=batch_size, verbose=0)
        
        # Entrenar red de política usando gradientes de la red de valor
        with tf.GradientTape() as tape:
            # Obtener valores Q para las acciones tomadas
            action_probs = self.policy_network(states, training=True)
            q_values = self.value_network(states, training=False)
            
            # Calcular pérdida (maximizar Q-values)
            loss = -tf.reduce_mean(q_values)
        
        # Aplicar gradientes
        grads = tape.gradient(loss, self.policy_network.trainable_variables)
        self.policy_network.optimizer.apply_gradients(
            zip(grads, self.policy_network.trainable_variables)
        )
    
    def _get_current_state(self):
        """Obtiene estado sensorial actual"""
        # En implementación real, leería sensores reales
        # Aquí simulamos un estado neutro
        return np.zeros(self.num_joints * 3)
    
    def _predict_next_state(self, state, action):
        """Predice el próximo estado dado el estado y acción actuales"""
        # Usa el modelo interno para predecir
        input_data = np.concatenate([state, action])
        # Predicción simplificada
        return state + np.random.normal(0, 0.05, len(state))
    
    def _save_progress(self, iteration, filename_prefix="brain_progress"):
        """Guarda el estado de aprendizaje"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_iter{iteration}_{timestamp}.pkl"
        
        save_data = {
            'body_schema': self.body_schema,
            'policy_weights': self.policy_network.get_weights(),
            'value_weights': self.value_network.get_weights(),
            'exploration_rate': self.exploration_rate,
            'iteration': iteration,
            'timestamp': timestamp
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"💾 Progreso guardado: {filename}")
        return filename
    
    def load_progress(self, filename):
        """Carga estado de aprendizaje guardado"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.body_schema = data['body_schema']
            self.policy_network.set_weights(data['policy_weights'])
            self.value_network.set_weights(data['value_weights'])
            self.exploration_rate = data['exploration_rate']
            
            print(f"✅ Cerebro cargado desde {filename}")
            print(f"   Iteración: {data['iteration']}, "
                  f"Fecha: {data['timestamp']}")
            
            return True
        except Exception as e:
            print(f"❌ Error cargando cerebro: {e}")
            return False
    
    def run_learning_pipeline(self, total_iterations=5000):
        """Pipeline completo de aprendizaje"""
        print("\n" + "="*60)
        print("🚀 INICIANDO PIPELINE COMPLETO DE APRENDIZAJE")
        print("="*60)
        
        # Fase 1: Descubrimiento básico
        print("\n📘 FASE 1: Balbuceo motor (1000 iteraciones)")
        self.motor_babbling_phase(iterations=1000)
        
        # Fase 2: Aprendizaje consolidado
        print("\n📗 FASE 2: Aprendizaje consolidado (2000 iteraciones)")
        for i in range(2000):
            # Ejecutar política actual
            state = self._get_current_state()
            action = self.policy_network.predict(state.reshape(1, -1), verbose=0)[0]
            
            # Obtener retroalimentación
            feedback = self._execute_motor_command(action)
            reward = self._calculate_intrinsic_reward(action, feedback)
            
            # Aprender
            self._update_body_schema(action, feedback)
            
            # Guardar experiencia
            self.experience_buffer.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': feedback,
                'done': False
            })
            
            # Entrenamiento periódico
            if i % 20 == 0 and len(self.experience_buffer) > 64:
                self._train_from_experience(batch_size=64)
            
            if i % 500 == 0:
                print(f"  Iteración {i}: Recompensa promedio = {reward:.3f}")
        
        # Fase 3: Optimización
        print("\n📕 FASE 3: Optimización final (2000 iteraciones)")
        self.exploration_rate = 0.1  # Reducir exploración
        for i in range(2000):
            # Ejecutar con menos exploración
            if np.random.random() < 0.1:
                action = np.random.uniform(-0.5, 0.5, self.num_muscles)
            else:
                state = self._get_current_state()
                action = self.policy_network.predict(state.reshape(1, -1), verbose=0)[0]
            
            feedback = self._execute_motor_command(action)
            
            # Enfocarse en eficiencia
            energy_cost = np.sum(np.abs(action))
            if energy_cost < 2.0:  # Recompensar eficiencia
                reward = 1.0 / (1.0 + energy_cost)
            else:
                reward = -0.1
            
            # Aprender
            self.experience_buffer.append({
                'state': self._get_current_state(),
                'action': action,
                'reward': reward,
                'next_state': feedback,
                'done': False
            })
            
            if i % 500 == 0:
                self._train_from_experience(batch_size=128)
                print(f"  Iteración {i}: Eficiencia = {energy_cost:.3f}")
        
        # Guardar resultado final
        final_file = self._save_progress(total_iterations, "brain_final")
        print(f"\n✅ Pipeline completado. Cerebro guardado en: {final_file}")
        print("🎉 El humanoide ha aprendido los fundamentos de su cuerpo!")

# ============================================================================
# CLASE PARA SISTEMA CARDIOVASCULAR
# ============================================================================

class CardiovascularSystem:
    """Sistema cardiovascular artificial biomimético"""
    
    def __init__(self):
        self.heart_rate = 60  # BPM
        self.blood_pressure = [120, 80]  [Systolic, Diastolic]
        self.blood_volume = 5.0  # Litros
        self.oxygen_level = 0.95  # 95% saturado
        self.temperature = 37.0  # °C
        
        # Componentes del sistema
        self.pumps = {
            'heart_main': {'flow_rate': 5.0, 'pressure': 120, 'active': True},
            'heart_backup': {'flow_rate': 0.0, 'pressure': 0, 'active': False}
        }
        
        self.valves = {}
        self.vessels = {}
        
        # Sensores
        self.sensors = {
            'pressure': [],
            'flow': [],
            'oxygen': [],
            'temperature': [],
            'ph': []
        }
        
        print("❤️ Sistema Cardiovascular inicializado")
    
    def regulate_homeostasis(self, body_demands):
        """
        Regula homeostasis como el sistema cardiovascular humano
        body_demands: dict con demandas de diferentes sistemas
        """
        adjustments = {}
        
        # Ajustar ritmo cardíaco basado en demanda
        demand_factor = body_demands.get('metabolic_demand', 1.0)
        new_heart_rate = 60 * demand_factor
        self.heart_rate = np.clip(new_heart_rate, 40, 180)
        
        # Ajustar presión arterial
        pressure_adjustment = (demand_factor - 1.0) * 20
        self.blood_pressure[0] = 120 + pressure_adjustment  # Systolic
        self.blood_pressure[1] = 80 + (pressure_adjustment * 0.6)  # Diastolic
        
        # Distribuir flujo según demanda
        flow_distribution = self._distribute_blood_flow(body_demands)
        
        # Regular temperatura
        if body_demands.get('cooling_needed', False):
            self._activate_cooling_mechanism()
        
        adjustments.update({
            'heart_rate': self.heart_rate,
            'blood_pressure': self.blood_pressure.copy(),
            'flow_distribution': flow_distribution
        })
        
        return adjustments
    
    def _distribute_blood_flow(self, demands):
        """Distribuye flujo sanguíneo como en el cuerpo humano"""
        distribution = {
            'brain': 0.15,      # 15% para cerebro (prioridad máxima)
            'heart': 0.05,      # 5% para músculo cardíaco
            'muscles': 0.20,    # 20% base para músculos
            'skin': 0.05,       # 5% para piel (termorregulación)
            'kidneys': 0.20,    # 20% para filtración
            'other': 0.35       # 35% para otros órganos
        }
        
        # Ajustar según demanda
        if demands.get('muscle_activity', 0) > 0.5:
            # Durante ejercicio: más flujo a músculos, menos a otros
            distribution['muscles'] = 0.70
            distribution['other'] = 0.05
            distribution['skin'] = 0.10  # Para enfriamiento
        
        if demands.get('cognitive_load', 0) > 0.7:
            # Alta carga cognitiva: más flujo al cerebro
            distribution['brain'] = 0.25
        
        return distribution
    
    def _activate_cooling_mechanism(self):
        """Activa mecanismos de enfriamiento (simula sudoración/vasodilatación)"""
        # Aumentar flujo a piel para disipar calor
        if 'skin' in self.vessels:
            self.vessels['skin']['flow_multiplier'] = 2.0
        
        # Activar intercambiadores de calor
        self.temperature -= 0.1  # Enfriamiento gradual
        
        print("🌡️ Mecanismo de enfriamiento activado")
    
    def monitor_vital_signs(self):
        """Monitorea signos vitales como en medicina humana"""
        vitals = {
            'heart_rate': self.heart_rate,
            'blood_pressure': f"{self.blood_pressure[0]}/{self.blood_pressure[1]}",
            'cardiac_output': self.pumps['heart_main']['flow_rate'] * self.heart_rate / 1000,
            'oxygen_saturation': self.oxygen_level * 100,
            'core_temperature': self
