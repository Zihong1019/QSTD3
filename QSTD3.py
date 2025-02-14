import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# Self-Attention-Based Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=1):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.attn1 = nn.MultiheadAttention(embed_dim=256, num_heads=num_heads)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.attn2 = nn.MultiheadAttention(embed_dim=256, num_heads=num_heads)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1).unsqueeze(0)

        # Q1 value estimation
        q1 = F.relu(self.l1(sa))
        q1, _ = self.attn1(q1, q1, q1)
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1).squeeze(0)

        # Q2 value estimation
        q2 = F.relu(self.l4(sa))
        q2, _ = self.attn2(q2, q2, q2)
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2).squeeze(0)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1).unsqueeze(0)
        q1 = F.relu(self.l1(sa))
        q1, _ = self.attn1(q1, q1, q1)
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1).squeeze(0)
        return q1


class QSTD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        num_heads=1
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, num_heads=num_heads).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, state, action, next_state, reward, not_done):
        self.total_it += 1

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

# Q-value-Based Sampling Method
    def sample_with_value(self, replay_buffer, batch_size, k):
        """
        Sample from the replay buffer based on the Q-value.
        """
        state, action, next_state, reward, not_done = replay_buffer.sample(k * batch_size)

        with torch.no_grad():
            value, _ = self.critic(state, action)
        value = value.squeeze()

        # Sort indices based on value
        sorted_indices = torch.argsort(value)
        third = len(value) // 3

        # Select indices for worst, middle, and best samples
        worst_indices = sorted_indices[:third]
        middle_indices = sorted_indices[third:2 * third]
        best_indices = sorted_indices[-third:]

        # Determine the number of samples to take from each group
        num_worst = batch_size // 3
        num_middle = batch_size // 3
        num_best = batch_size - num_worst - num_middle

        # Mixed sampling
        worst_sample_indices = np.random.choice(worst_indices.cpu().numpy(), num_worst, replace=False)
        middle_sample_indices = np.random.choice(middle_indices.cpu().numpy(), num_middle, replace=False)
        best_sample_indices = np.random.choice(best_indices.cpu().numpy(), num_best, replace=False)
        sample_indices = np.concatenate([best_sample_indices, middle_sample_indices, worst_sample_indices])

        return (
            state[sample_indices],
            action[sample_indices],
            next_state[sample_indices],
            reward[sample_indices],
            not_done[sample_indices]
        )