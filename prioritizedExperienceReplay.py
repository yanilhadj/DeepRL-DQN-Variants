from collections import deque
import numpy as np
import torch 

class PrioritizedExperienceReplay:
    """
    Implémente une mémoire avec priorités pour l'expérience replay.
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Initialisation de la mémoire d'expérience avec priorité.
        
        Args:
            capacity : Taille maximale de la mémoire.
            alpha: Contrôle l'importance des priorités (0 = échantillonnage uniforme).
            beta_start: Valeur initiale pour la correction d'importance.
            beta_frames: Nombre de frames pour atteindre la valeur maximale de beta (1).
        """
        self.buffer = deque([], maxlen=capacity)  # Utilisation d'une deque pour stocker les expériences
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Priorités associées aux expériences
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.capacity = capacity
        self.position = 0
        self.frame = 1

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        """
        Ajoute une expérience dans la mémoire.
        
        experience est un tuple : (state, action, reward, next_state, done).
        """
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Tire un échantillon pondéré par les priorités.
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        beta = self.beta_start + (1 - self.beta_start) * (self.frame / self.beta_frames)
        beta = min(1.0, beta)

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        self.frame += 1

        states, actions, rewards, next_states, dones = zip(*experiences)

        return (
          np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in states]),  # États
          np.array(actions, dtype=np.int64),  # Actions
          np.array(rewards, dtype=np.float32),  # Récompenses
          np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in next_states]),  # Nouveaux états
          np.array(dones, dtype=bool)), np.array(weights, dtype=np.float32), np.array(indices, dtype=np.int64)  # Indices


    def update_priorities(self, indices, priorities):
        """
        Met à jour les priorités des expériences échantillonnées.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
