from collections import deque
import numpy as np
import torch 

class ExperienceReplay: 
    """
    Nous avons besoin d'un grand nombre de simulation d'episode afin de "feed" le réseau de neurone.
    Nous allons utilisé une méthode nommée "Experience Replay" (cours 5 page 26).
    Une experience est constituée de 5 champs : 
        1- L'état (s)     2- L'action (a)       3- La récompense (r)         4- Le nouvel état (s')       5- L'état du jeu (fini ou pas fini)
    Nous allons sauvegardé cette combinaison dans une "Python Deque" qui est une liste optimisée pour l'ajout et la suppression 
    d'éléments à ses deux extrémités (on peut ajouter et supprimer deux éléments à la fois en temps constant O(1).
    Cela va permettre d'avoir une gestion optimale de la mémoire.
    """

    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)  # Utilisation d'une deque pour stocker les expériences

    def __len__(self):
        # Retourner la longueur actuelle du buffer
        return len(self.buffer)

    def append(self, experience):
        # Ajouter une expérience (état, action, récompense, fait, prochain état)
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Prélever un échantillon aléatoire dans le buffer
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # Décompresser les expériences sélectionnées
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
          np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in states]),  # États
          np.array(actions, dtype=np.int64),  # Actions
          np.array(rewards, dtype=np.float32),  # Récompenses
          np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in next_states]),  # Nouveaux états
          np.array(dones, dtype=bool)
        )


