import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class qNetwork(nn.Module):
    def __init__(self, input_shape, action_dim):
        """
        Description :
            Réseau de neurones générique pour estimer la fonction Q(s, a).
        Paramètres :
            input_shape : tuple représentant la forme de l'entrée (ex. : (channels, hauteur, largeur) ou (2,))
            action_dim : dimension de l'espace des actions.
        """
        super(qNetwork, self).__init__()
        self.sequentiel = False

        if len(input_shape) == 1:  # Cas d'une entrée vectorielle
            self.sequentiel = True
            self.fc1 = nn.Linear(input_shape[0], 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, action_dim)

        else:  # Cas d'une entrée convolutive (image)
            channels, height, width = input_shape

            self.Conv1 = nn.Conv2d(channels, 32, 8, stride=4)
            self.Conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.Conv3 = nn.Conv2d(64, 64, 3, stride=1)

            def conv2d_size_out(size, kernel_size, stride):
                return (size - kernel_size) // stride + 1

            conv1_h = conv2d_size_out(height, 8, 4)
            conv1_w = conv2d_size_out(width, 8, 4)
            conv2_h = conv2d_size_out(conv1_h, 4, 2)
            conv2_w = conv2d_size_out(conv1_w, 4, 2)
            conv3_h = conv2d_size_out(conv2_h, 3, 1)
            conv3_w = conv2d_size_out(conv2_w, 3, 1)
            linear_input_size = conv3_h * conv3_w * 64

            self.Linear1 = nn.Linear(linear_input_size, 512)
            self.Linear2 = nn.Linear(512, action_dim)

    def forward(self, x):
        if self.sequentiel:  # Cas vectorielc
            """Build a network that maps state -> action values."""
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            return self.fc3(x)
        
        else:  # Cas convolutif
            x = F.relu(self.Conv1(x))
            x = F.relu(self.Conv2(x))
            x = F.relu(self.Conv3(x))
            x = torch.flatten(x, 1)
            x = F.relu(self.Linear1(x))
            return self.Linear2(x)

    @staticmethod
    def save_model(policy_dqn, optimizer, episode, total_time, filepath):
        """
        Sauvegarde l'état actuel du modèle DQN, de l'optimiseur, de l'épisode et du temps total d'exécution.
        """
        checkpoint = {
            "model_state_dict": policy_dqn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "total_time": total_time
        }
        torch.save(checkpoint, filepath)
        print(f"Modele sauvegarde a l episode {episode}, Temps total d execution : {total_time:.2f} secondes")


    @staticmethod
    def load_model(policy_dqn, optimizer, filepath):
        """
        Charge l'état du modèle DQN, de l'optimiseur, de l'épisode et du temps total d'exécution.
        """
        checkpoint = torch.load(filepath)
        policy_dqn.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        episode = checkpoint["episode"]
        total_time = checkpoint.get("total_time", 0)  # Utilisation de 0 par défaut si "total_time" n'est pas trouvé
        print(f"Modele charge depuis l episode {episode} Temps total : {total_time:.2f} secondes")
        return episode, total_time