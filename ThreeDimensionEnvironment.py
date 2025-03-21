import gymnasium as gym
from qNetwork import qNetwork
import torch
from experience_replay import ExperienceReplay
import yaml
import random
import time
import os
import matplotlib.pyplot as plt  # Pour tracer les récompenses
import numpy as np
# Détection de l'appareil : Utilisation du GPU si disponible
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, game_name, input_shape, action_dim, render=True):
        """
        Description : 
            Permet d'initialiser 
        """
        # On récupère les paramètres
        with open('hyperparameters.yml', 'r') as file:
            all_parameters = yaml.safe_load(file)
            self.hyperparameters = all_parameters[game_name]
        # Hyperparamètres
        self.replay_memory_size = self.hyperparameters["replay_memory_size"]
        self.mini_batch_size = self.hyperparameters["mini_batch_size"]
        self.epsilon_init = self.hyperparameters["epsilon_init"]
        self.epsilon_decay = self.hyperparameters["epsilon_decay"]
        self.epsilon_min = self.hyperparameters["epsilon_min"]
        self.gamma = self.hyperparameters["gamma"]
        self.learning_rate = self.hyperparameters["learning_rate"]
        
        # Information relative à l'environnement
        self.game_name = game_name
        self.input_shape = input_shape
        self.action_dim = action_dim
        self.env = gym.make(game_name, render_mode="human" if render else None)

    def select_action(self, state, policy_qNetwork, epsilon):
        if random.random() < epsilon:
            # Si la probabilité est inférieur à epsilon on pioche une action aléatoire 
            # Exploration
            action = self.env.action_space.sample()
            return action
        else: 
            with torch.no_grad():
                # En utilisant le réseau de neurone on estime la valeur de q(state, a)
                # pour chaque action possible a.
                # On choisit l'action qui maximise cette estimation
                action = policy_qNetwork(state.unsqueeze(0)).argmax(dim=1).item()
                return action

    def train_step(self, memory, policy_qNetwork, target_qNetwork, optimizer):
        """
        Effectue une mise à jour des paramètres Q-Learning.
        Si : policy_qNetwork != target_qNetwork => Double DQ-network
        """
        states, actions, rewards, next_states, dones = memory.sample(self.mini_batch_size)
        # Conversion en tensors
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        # Dimensions de next_states : (taille_batch, nombre_cannaux, hauteur, largeur )
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)

        # Calcul de la cible : y_i = r + \gamma max_a' Q(s', a')
        with torch.no_grad():
            # on calcule q(s′i ,a", w)
            q_next = policy_qNetwork(next_states).max(dim=1)[0]
            # 1- dones.float() : permet de ne pas prendre en considération
            q_target = rewards + (1 - dones.float()) * self.gamma * q_next

        # Calcul de la prédiction Q(s, a)
        # Calcul de la prédiction Q(s, a)
        q_pred = policy_qNetwork(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Calcul de la perte
        loss = torch.nn.functional.mse_loss(q_pred, q_target)

        # Optimisation du modèle
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def DqNetwork_experienceReplay(self, max_episodes, qNetwork_file,is_training=True, render=True):
        # Initialisation du réseau qNetwork et de la mémoire d'expérience
        policy_qNetwork = qNetwork(self.input_shape, self.action_dim).to(device)
        optimizer = torch.optim.RMSprop(policy_qNetwork.parameters(), lr=self.learning_rate, alpha=0.95, eps=1e-7)
        memory = ExperienceReplay(self.replay_memory_size)
        
        rewards_per_episodes, episode_times = [], []
        start_episode, total_time = 0, 0  
        epsilon = self.epsilon_init

        if os.path.exists(qNetwork_file):
            start_episode, total_time = qNetwork.load_model(policy_qNetwork, optimizer, qNetwork_file)

        # Boucle sur les épisodes
        for episode in range(start_episode, start_episode+max_episodes):
            episode_start_time = time.time()
            # Réinitialisation de l'environnement
            # state : représente l'état initial de l'environnement
            state, _ = self.env.reset()
            # on transforme l'état en un tenseur torch 
            state = torch.tensor(state, dtype=torch.float32, device=device)  
            state = state.permute(2, 0, 1)  # Réarrangement en [channels, height, width]
            terminated = False
            episode_reward, episode_start_time = 0, time.time()

            # Boucle sur les étapes de l'épisode (tant que l'épisode n'est pas achevé)
            while not terminated:
                # Choix de l'action selon la politique epsilon-greedy
                action = self.select_action(state, policy_qNetwork, epsilon)

                # on execute l'action a choisie en étant dans l'état state 
                new_state, reward, terminated, _, info = self.env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                new_state = new_state.permute(2, 0, 1)  # Réarrangement en [batch_size, channels, height, width]
                # Mise à jour des récompenses accumulées
                episode_reward += reward

                # enregistrer la transition (st,at,rt+1,st+1) dans la mémoire D
                memory.append((state, action, reward, new_state, terminated))

                # tirer des transitions (s,a,r,s′) ∈ D (mini batch)
                if len(memory) >= self.mini_batch_size :
                    self.train_step(memory,policy_qNetwork, policy_qNetwork, optimizer)                
                state = new_state

            # FIN EPISODE
            rewards_per_episodes.append(episode_reward)
            episode_times.append(time.time() - episode_start_time)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)  # Décroissance d'epsilon
            print(f"Épisode {episode} - Récompense : {episode_reward:.2f} - Epsilon : {epsilon:.3f}")
            # Sauvegarde périodique du modèle
            
            if episode % 5 == 0 :
                qNetwork.save_model(policy_qNetwork, optimizer, episode, total_time, qNetwork_file)

        return rewards_per_episodes, episode_times
    
    def soft_update(self, local_model, target_model, tau=1e-3):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def DoubleDqNetwork_experienceReplay(self, max_episodes, qNetwork_file, render=True):
        # Initialisation du réseau qNetwork et de la mémoire d'expérience
        policy_qNetwork = qNetwork(self.input_shape, self.action_dim).to(device)
        target_qNetwork = qNetwork(self.input_shape, self.action_dim).to(device)  # Réseau cible
        optimizer = torch.optim.RMSprop(policy_qNetwork.parameters(), lr=self.learning_rate, alpha=0.95, eps=1e-7)
        memory = ExperienceReplay(self.replay_memory_size)
        
        rewards_per_episodes, episode_times = [], []
        start_episode, total_time = 0, 0  
        epsilon = self.epsilon_init

        if os.path.exists(qNetwork_file):
            start_episode, total_time = qNetwork.load_model(policy_qNetwork, optimizer, qNetwork_file)

        # Boucle sur les épisodes
        for episode in range(start_episode, start_episode+max_episodes):
            episode_start_time = time.time()
            # Réinitialisation de l'environnement
            # state : représente l'état initial de l'environnement
            state, _ = self.env.reset()
            # on transforme l'état en un tenseur torch 
            state = torch.tensor(state, dtype=torch.float32, device=device)  
            state = state.permute(2, 0, 1)  # Réarrangement en [channels, height, width]
            terminated = False
            episode_reward, episode_start_time = 0, time.time()

            # Boucle sur les étapes de l'épisode (tant que l'épisode n'est pas achevé)
            while not terminated:
                # Choix de l'action selon la politique epsilon-greedy
                action = self.select_action(state, policy_qNetwork, epsilon)

                # on execute l'action a choisie en étant dans l'état state 
                new_state, reward, terminated, _, info = self.env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                new_state = new_state.permute(2, 0, 1)  # Réarrangement en [batch_size, channels, height, width]
                # Mise à jour des récompenses accumulées
                episode_reward += reward

                # enregistrer la transition (st,at,rt+1,st+1) dans la mémoire D
                memory.append((state, action, reward, new_state, terminated))

                # tirer des transitions (s,a,r,s′) ∈ D (mini batch)
                if len(memory) >= self.mini_batch_size :
                    self.train_step(memory,policy_qNetwork, target_qNetwork, optimizer) 
                    self.soft_update(policy_qNetwork, target_qNetwork)

                state = new_state

            # FIN EPISODE
            rewards_per_episodes.append(episode_reward)
            episode_times.append(time.time() - episode_start_time)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)  # Décroissance d'epsilon
            print(f"Épisode {episode} - Récompense : {episode_reward:.2f} - Epsilon : {epsilon:.3f}")
            # Sauvegarde périodique du modèle
            
            if episode % 5 == 0 :
                qNetwork.save_model(policy_qNetwork, optimizer, episode, total_time, qNetwork_file)

            return rewards_per_episodes, episode_times
    
agent = Agent("ALE/Pong-v5", input_shape=(3, 210, 160), action_dim=6)
#rewards_per_episodes, episode_time = agent.DqNetwork_experienceReplay(max_episodes=5, qNetwork_file="qNetwork_checkpoint.pth")
rewards_per_episodes, episode_time = agent.DoubleDqNetwork_experienceReplay(max_episodes=6, qNetwork_file="DoubleqNetwork_checkpoint.pth")
# Tracé des récompenses par épisode
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episodes, label="Recompense par episode")
plt.xlabel("Episodes")
plt.ylabel("Recompense")
plt.title("Evolution des recompenses")
plt.legend()

# Tracé des temps par épisode
plt.subplot(1, 2, 2)
plt.plot(episode_time, label="Temps par episode")
plt.xlabel("Episodes")
plt.ylabel("Temps (secondes)")
plt.title("Temps par episode")
plt.legend()

plt.tight_layout()
plt.show()