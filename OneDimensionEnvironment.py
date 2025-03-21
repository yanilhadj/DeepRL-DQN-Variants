import gymnasium as gym
import torch
import yaml
import random
import time
import os
import math
import matplotlib.pyplot as plt
from qNetwork import qNetwork
from experience_replay import ExperienceReplay
import numpy as np
from prioritizedExperienceReplay import PrioritizedExperienceReplay
import torch.optim as optim
from collections import deque

# Détection automatique du GPU ou CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Appareil utilisé : {device}")

class Agent:
    def __init__(self, game_name, input_shape, action_dim, render=True):
        """
        Description :
            Permet d'initialiser les différents paramètres qui définissent l'agent et l'environnement
        """
        # On récupère les paramètres
        with open('Projet/hyperparameters.yml', 'r') as file:
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
        """Sélectionne une action avec epsilon-greedy."""
        # On transforme l'etat en tensor et ajoute une dimension (spécification torch)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Passe en mode évaluation pour des prédictions stables
        policy_qNetwork.eval()

        # On récupère les valeurs associées à chaque pair (state, action)
        with torch.no_grad():
            action_values = policy_qNetwork(state)

        # Retour au mode entraînement pour les futurs calculs
        policy_qNetwork.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().numpy())
        else:
            return self.env.action_space.sample()


    def train_step(self, memory, policy_qNetwork, target_qNetwork, optimizer, prioritized=False):
        """
        Effectue une mise à jour des paramètres Q-Learning.
        Si : policy_qNetwork != target_qNetwork => Double DQ-network
        """
        # On pioche depuis la mémoire
        if prioritized :
            (states, actions, rewards, next_states, dones), weights, indices = memory.sample(self.mini_batch_size)
        else :
            states, actions, rewards, next_states, dones = memory.sample(self.mini_batch_size)

        # Conversion en tensors
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)

        if prioritized :
            weights = torch.tensor(weights, dtype=torch.float32, device=device)

        # Calcul de Q(s', a') et Q(s, a)
        with torch.no_grad():
            # Actions optimales sélectionnées par le réseau
            best_actions = policy_qNetwork(next_states).argmax(dim=1, keepdim=True)
            # Valeurs Q(s',_) correspondantes à ces actions sélectionnées dans le réseau cible
            # q_next = policy_qNetwork(next_states).gather(1, best_actions).squeeze(1)
            q_next = target_qNetwork(next_states).gather(1, best_actions).squeeze(1)
            q_target = rewards + (1 - dones.float()) * self.gamma * q_next

        q_pred = policy_qNetwork(states).gather(1, actions).squeeze(1)

        # Mise à jour du modèle
        if prioritized :
            loss = (weights * (q_pred - q_target) ** 2).mean()
        else :
            loss = torch.nn.functional.mse_loss(q_pred, q_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if prioritized :
            # Calcul des erreurs TD pour mettre à jour les priorités
            td_errors = torch.abs(q_pred - q_target).detach().cpu().numpy()
            memory.update_priorities(indices, td_errors)

    def plot_results(self, rewards, episode_times, window_size=10):
            if len(rewards) < window_size:
                print("Pas assez d'épisodes pour une moyenne glissante, affichage des données brutes.")
                window_size = len(rewards)

            episodes = np.arange(len(rewards))

            # Calcul de la moyenne et de la variance glissante
            rewards_mean = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            rewards_var = [np.var(rewards[max(0, i - window_size + 1):i + 1]) for i in range(len(rewards))]
            episodes_mean = episodes[:len(rewards_mean)]  # Ajustement pour l'alignement

            plt.figure(figsize=(12, 6))

            # Graphique des récompenses
            plt.subplot(1, 2, 1)
            plt.plot(episodes, rewards, label="Récompense par épisode", alpha=0.5)
            plt.plot(episodes_mean, rewards_mean, label="Moyenne glissante", color='red')
            plt.xlabel("Épisodes")
            plt.ylabel("Récompense")
            plt.legend()
            plt.title("Évolution des récompenses")

            # Graphique de la variance des récompenses
            plt.subplot(1, 2, 2)
            plt.plot(episodes, rewards_var, label="Variance des récompenses", color='blue')
            plt.xlabel("Épisodes")
            plt.ylabel("Variance")
            plt.legend()
            plt.title("Variance des récompenses sur 10 épisodes")

            plt.tight_layout()
            plt.show()

    # =================================================== Deep Q-networks Experience Replay =========================================================================================
    def DqNetwork_experienceReplay(self, max_episodes, qNetwork_file):
        # Initialisation
        policy_qNetwork = qNetwork(self.input_shape, self.action_dim).to(device)
        optimizer = torch.optim.Adam(policy_qNetwork.parameters(), lr=self.learning_rate)
        memory = ExperienceReplay(self.replay_memory_size)

        rewards_per_episodes, episode_times = [], []
        start_episode, total_time = 0, 0
        epsilon = self.epsilon_init

        # Si les paramètres du modèle ont déja étaient save on les récupère
        if os.path.exists(qNetwork_file):
            start_episode, total_time = qNetwork.load_model(policy_qNetwork, optimizer, qNetwork_file)

        # On simule des épisodes en entier
        for episode in range(start_episode, start_episode + max_episodes):

            # On récupère l'état courant
            state, _ = self.env.reset()
            episode_reward, episode_start_time = 0, time.time()
            terminated = False

            while not terminated:
                # On séléctionne l'action à effectuer
                action = self.select_action(state, policy_qNetwork, epsilon)
                # On récupère le nouvel état de l'envronnement après avoir exécuter l'action
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                terminated = terminated or truncated
                episode_reward += reward

                memory.append((state, action, reward, new_state, terminated))

                if len(memory) >= self.mini_batch_size:
                    self.train_step(memory, policy_qNetwork, policy_qNetwork, optimizer)
  
                state = new_state

            # Fin d'épisode
            rewards_per_episodes.append(episode_reward)
            episode_times.append(time.time() - episode_start_time)
            # mise à jour de epsilon
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)  # Décroissance d'epsilon
            print(f"Épisode {episode} - Récompense : {episode_reward:.2f} - Epsilon : {epsilon:.3f}")

            if episode % 5 == 0 :
                qNetwork.save_model(policy_qNetwork, optimizer, episode, total_time, qNetwork_file)
        return rewards_per_episodes, episode_times
    


  # =================================================== Deep Q-networks Prioritized Experience Replay=========================================================================================

    def DqNetwork_PrioritizedExperienceReplay(self, max_episodes, qNetwork_file):
        # Initialisation
        policy_qNetwork = qNetwork(self.input_shape, self.action_dim).to(device)
        optimizer = torch.optim.Adam(policy_qNetwork.parameters(), lr=self.learning_rate)
        memory = PrioritizedExperienceReplay(self.replay_memory_size)

        rewards_per_episodes, episode_times = [], []
        start_episode, total_time = 0, 0
        epsilon = self.epsilon_init

        # Si les paramètres du modèle ont déja étaient save on les récupère
        if os.path.exists(qNetwork_file):
            start_episode, total_time = qNetwork.load_model(policy_qNetwork, optimizer, qNetwork_file)

        # On simule des épisodes en entier
        for episode in range(start_episode, start_episode + max_episodes):

            # On récupère l'état courant
            state, _ = self.env.reset()
            episode_reward, episode_start_time = 0, time.time()
            terminated = False

            while not terminated:
                # On séléctionne l'action à effectuer
                action = self.select_action(state, policy_qNetwork, epsilon)
                # On récupère le nouvel état de l'envronnement après avoir exécuter l'action
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                terminated = terminated or truncated
                episode_reward += reward

                memory.append((state, action, reward, new_state, terminated))

                if len(memory) >= self.mini_batch_size:
                    self.train_step(memory, policy_qNetwork, policy_qNetwork, optimizer, prioritized=True)

                state = new_state

            # Fin d'épisode
            rewards_per_episodes.append(episode_reward)
            episode_times.append(time.time() - episode_start_time)
            # mise à jour de epsilon
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)  # Décroissance d'epsilon
            print(f"Épisode {episode} - Récompense : {episode_reward:.2f} - Epsilon : {epsilon:.3f}")

            if episode % 5 == 0 :
                qNetwork.save_model(policy_qNetwork, optimizer, episode, total_time, qNetwork_file)


        return rewards_per_episodes, episode_times

  # =================================================== Double Deep Q-networks Experience Replay=========================================================================================
    def soft_update(self, local_model, target_model, tau=1e-2):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model : weights will be copied from
            target_model : weights will be copied to
            tau : interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def DoubleDqNetwork_experienceReplay(self, max_episodes, qNetwork_file):
        # Initialisation
        policy_qNetwork = qNetwork(self.input_shape, self.action_dim).to(device)  # Réseau local
        target_qNetwork = qNetwork(self.input_shape, self.action_dim).to(device)  # Réseau cible
        optimizer = torch.optim.Adam(policy_qNetwork.parameters(), lr=self.learning_rate)

        memory = ExperienceReplay(self.replay_memory_size)

        rewards_per_episodes, episode_times = [], []
        start_episode, total_time = 0, 0
        epsilon = self.epsilon_init

        # Charger le modèle si disponible
        if os.path.exists(qNetwork_file):
            start_episode, total_time = qNetwork.load_model(policy_qNetwork, optimizer, qNetwork_file)

        # Simulation des épisodes
        for episode in range(start_episode, start_episode + max_episodes):

            # État initial
            state, _ = self.env.reset()
            episode_reward, episode_start_time = 0, time.time()
            terminated = False

            while not terminated:
                # Sélection de l'action (epsilon-greedy)
                action = self.select_action(state, policy_qNetwork, epsilon)
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                terminated = terminated or truncated
                episode_reward += reward

                # Stockage de la transition
                memory.append((state, action, reward, new_state, terminated))

                # Entraînement si suffisamment de transitions sont collectées
                if len(memory) >= self.mini_batch_size:
                    self.train_step(memory, policy_qNetwork, target_qNetwork, optimizer)
                    self.soft_update(policy_qNetwork, target_qNetwork)

                state = new_state

            # Fin d'épisode
            rewards_per_episodes.append(episode_reward)
            episode_times.append(time.time() - episode_start_time)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)  # Mise à jour epsilon
            print(f"Épisode {episode} - Récompense : {episode_reward:.2f} - Epsilon : {epsilon:.3f}")

            # Mise à jour du réseau cible tous les 5 épisodes
            if episode % 5 == 0 :
                qNetwork.save_model(policy_qNetwork, optimizer, episode, total_time, qNetwork_file)


        return rewards_per_episodes, episode_times
    
  # =================================================== Double Deep Q-networks Prioritized Experience Replay=========================================================================================

    def DoubleDqNetwork_PrioritizedExperienceReplay(self, max_episodes, qNetwork_file):
        # Initialisation
        policy_qNetwork = qNetwork(self.input_shape, self.action_dim).to(device)  # Réseau local
        target_qNetwork = qNetwork(self.input_shape, self.action_dim).to(device)  # Réseau cible
        optimizer = torch.optim.Adam(policy_qNetwork.parameters(), lr=self.learning_rate)

        memory = PrioritizedExperienceReplay(self.replay_memory_size)

        rewards_per_episodes, episode_times = [], []
        start_episode, total_time = 0, 0
        epsilon = self.epsilon_init

        # Charger le modèle si disponible
        if os.path.exists(qNetwork_file):
            start_episode, total_time = qNetwork.load_model(policy_qNetwork, optimizer, qNetwork_file)

        # Simulation des épisodes
        for episode in range(start_episode, start_episode + max_episodes):

            # État initial
            state, _ = self.env.reset()
            episode_reward, episode_start_time = 0, time.time()
            terminated = False

            while not terminated:
                # Sélection de l'action (epsilon-greedy)
                action = self.select_action(state, policy_qNetwork, epsilon)
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                terminated = terminated or truncated
                episode_reward += reward

                # Stockage de la transition
                memory.append((state, action, reward, new_state, terminated))

                # Entraînement si suffisamment de transitions sont collectées
                if len(memory) >= self.mini_batch_size:
                    self.train_step(memory, policy_qNetwork, target_qNetwork, optimizer, prioritized=True)
                    self.soft_update(policy_qNetwork, target_qNetwork)

                state = new_state

            # Fin d'épisode
            rewards_per_episodes.append(episode_reward)
            episode_times.append(time.time() - episode_start_time)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)  # Mise à jour epsilon
            print(f"Épisode {episode} - Récompense : {episode_reward:.2f} - Epsilon : {epsilon:.3f}")

            # Mise à jour du réseau cible tous les 5 épisodes
            if episode % 5 == 0 :
                qNetwork.save_model(policy_qNetwork, optimizer, episode, total_time, qNetwork_file)


        return rewards_per_episodes, episode_times

    

def main():
    print("Choisissez un environnement :")
    environments = {
        "1": ("MountainCar-v0", (2,), 3)
        #,"2": ("LunarLander-v3", (8,), 4)
    }
    for key, (name, _, _) in environments.items():
        print(f"{key}. {name}")

    env_choice = input("Entrez le numéro de l'environnement : ").strip()
    if env_choice not in environments:
        print("Choix invalide. Veuillez relancer le script.")
        return

    env_name, input_shape, action_dim = environments[env_choice]

    print("\nChoisissez l'algorithme :")
    algorithms = {
        "1": ("DQN", "dqn.pth"),
        "2": ("DQN-PER", "prioritized_dqn.pth"),
        "3": ("DoubleDQN", "double_dqn.pth"),
        "4": ("DoubleDQN-PER", "prioritized_double_dqn.pth")
    }
    for key, (algo, _) in algorithms.items():
        print(f"{key}. {algo}")

    algo_choice = input("Entrez le numéro de l'algorithme : ").strip()
    if algo_choice not in algorithms:
        print("Choix invalide. Veuillez relancer le script.")
        return

    algorithm, model_file = algorithms[algo_choice]

    # Initialisation de l'agent
    agent = Agent(env_name, input_shape, action_dim)

    # Exécution de l'algorithme sans appel direct à train
    max_episodes = 1000
    if algorithm == "DQN":
        results = agent.DqNetwork_experienceReplay(max_episodes, model_file)
    elif algorithm == "DQN-PER":
        results = agent.DqNetwork_PrioritizedExperienceReplay(max_episodes, model_file)
    elif algorithm == "DoubleDQN":
        results = agent.DoubleDqNetwork_experienceReplay(max_episodes, model_file)
    elif algorithm == "DoubleDQN-PER":
        results = agent.DoubleDqNetwork_PrioritizedExperienceReplay(max_episodes, model_file)

    # Affichage des résultats
    rewards, episode_times = results
    agent.plot_results(rewards, episode_times)

if __name__ == "__main__":
    main()
