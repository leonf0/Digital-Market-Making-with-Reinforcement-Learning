import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque

@dataclass
class MarketConfig:
    T: float = 1.0
    dt: float = 0.005
    sigma: float = 0.5
    start_price: float = 100.0
    k: float = 1.5
    A: float = 140

class MarketEnv:
    def __init__(self, config: MarketConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.curr_time = 0
        self.fair_price = self.config.start_price
        self.price_history = [self.fair_price]
        self.time_history = [0]
        return self.fair_price

    def step_price(self):
        dt, sigma = self.config.dt, self.config.sigma
        z = np.random.normal()
        self.fair_price *= np.exp(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z)
        self.curr_time += dt
        self.price_history.append(self.fair_price)
        self.time_history.append(self.curr_time)
        return self.fair_price

    def execute_orders(self, bid, ask):
        bid = min(bid, self.fair_price)
        ask = max(ask, self.fair_price)
        dt_bid = self.fair_price - bid
        dt_ask = ask - self.fair_price
        lambda_bid = self.config.A * np.exp(-self.config.k * dt_bid)
        lambda_ask = self.config.A * np.exp(-self.config.k * dt_ask)
        p_bid = np.clip(lambda_bid * self.config.dt, 0.0, 1.0)
        p_ask = np.clip(lambda_ask * self.config.dt, 0.0, 1.0)
        bid_filled = np.random.random() < p_bid
        ask_filled = np.random.random() < p_ask
        return bid_filled, ask_filled

class MarketMaker:
    def __init__(self):
        self.cash = 0
        self.inventory = 0
        self.inventory_history = []
        self.wealth_history = []

    def reset(self):
        self.cash = 0
        self.inventory = 0
        self.inventory_history = []
        self.wealth_history = []

    def update_state(self, fair_price, bid_filled, ask_filled, bid_price, ask_price,
                     maker_rebate_pct=0.0002, taker_fee_pct=0.0005):
        if bid_filled:
            self.inventory += 1
            self.cash -= bid_price * (1 - maker_rebate_pct)
        if ask_filled:
            self.inventory -= 1
            self.cash += ask_price * (1 + maker_rebate_pct)
        curr_wealth = self.cash + self.inventory * fair_price
        self.wealth_history.append(curr_wealth)
        self.inventory_history.append(self.inventory)
        return curr_wealth

class NaiveMarketMaker(MarketMaker):
    def __init__(self, spread=0.5):
        super().__init__()
        self.spread = spread

    def get_quotes(self, curr_time, fair_price):
        half = self.spread / 2
        return fair_price - half, fair_price + half

class AvellanedaStoikov(MarketMaker):
    def __init__(self, T, sigma, gamma, k):
        super().__init__()
        self.T = T
        self.sigma = sigma
        self.gamma = gamma
        self.kappa = k

    def reservation_price(self, fair_price, t):
        time_remaining = max(0, self.T - t)
        return fair_price - self.inventory * self.gamma * self.sigma**2 * time_remaining

    def optimal_spread(self, t):
        time_remaining = max(0, self.T - t)
        term1 = self.gamma * self.sigma**2 * time_remaining / 2
        term2 = (2 / self.gamma) * np.log(1 + self.gamma / self.kappa)
        return term1 + term2

    def get_quotes(self, curr_time, fair_price):
        mid = self.reservation_price(fair_price, curr_time)
        half = self.optimal_spread(curr_time) / 2
        return mid - half, mid + half

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.actor_spread = nn.Linear(32, 3)
        self.actor_skew = nn.Linear(32, 3)
        self.critic = nn.Linear(32, 1)

    def forward(self, state):
        features = self.shared(state)
        spread_logits = self.actor_spread(features)
        skew_logits = self.actor_skew(features)
        value = self.critic(features)
        return spread_logits, skew_logits, value

    def get_action_and_value(self, state):
        spread_logits, skew_logits, value = self.forward(state)

        spread_dist = torch.distributions.Categorical(logits=spread_logits)
        skew_dist = torch.distributions.Categorical(logits=skew_logits)

        spread_action = spread_dist.sample()
        skew_action = skew_dist.sample()

        log_prob = spread_dist.log_prob(spread_action) + skew_dist.log_prob(skew_action)
        entropy = spread_dist.entropy() + skew_dist.entropy()

        return spread_action, skew_action, log_prob, entropy, value

    def evaluate_actions(self, state, spread_action, skew_action):
        spread_logits, skew_logits, value = self.forward(state)

        spread_dist = torch.distributions.Categorical(logits=spread_logits)
        skew_dist = torch.distributions.Categorical(logits=skew_logits)

        log_prob = spread_dist.log_prob(spread_action) + skew_dist.log_prob(skew_action)
        entropy = spread_dist.entropy() + skew_dist.entropy()

        return log_prob, entropy, value

class RLMarketMaker(MarketMaker):
    def __init__(self, spread_delta=0.1, skew_delta=0.1, inventory_risk_aversion=1.0, profit_incentive=10):
        super().__init__()
        self.spread_delta = spread_delta
        self.skew_delta = skew_delta
        self.inventory_risk_aversion = inventory_risk_aversion
        self.profit_incentive = profit_incentive
        self.initial_spread = 0.5
        self.current_spread = 0.5
        self.current_skew = 0.0

    def reset(self):
        super().reset()
        self.current_spread = self.initial_spread
        self.current_skew = 0.0

    def get_quotes(self, curr_time, fair_price):
        mid = fair_price + self.current_skew
        half = self.current_spread / 2
        return mid - half, mid + half

    def adjust_spread(self, action):
        spread_action, skew_action = action
        if spread_action == 0:
            self.current_spread -= self.spread_delta
        elif spread_action == 2:
            self.current_spread += self.spread_delta
        self.current_spread = np.clip(self.current_spread, 0.1, 2.0)

        if skew_action == 0:
            self.current_skew -= self.skew_delta
        elif skew_action == 2:
            self.current_skew += self.skew_delta
        self.current_skew = np.clip(self.current_skew, -0.5, 0.5)

def choose_ppo_action(model, obs):
    state = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        spread_action, skew_action, log_prob, entropy, value = model.get_action_and_value(state)
    return (spread_action.item(), skew_action.item()), log_prob.item(), value.item()

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

def run_episode(model, env, agent, config):
    env.reset()
    agent.reset()

    states = []
    spread_actions = []
    skew_actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []

    steps = int(config.T / config.dt)
    prev_price = config.start_price
    prev_inventory = 0

    time_remaining = (config.T - env.curr_time) / config.T
    obs = [agent.inventory, time_remaining, agent.current_spread, agent.current_skew, 0, 0]

    for step in range(steps):
        states.append(obs)

        action, log_prob, value = choose_ppo_action(model, obs)
        spread_actions.append(action[0])
        skew_actions.append(action[1])
        log_probs.append(log_prob)
        values.append(value)

        agent.adjust_spread(action)

        prev_price = env.fair_price
        fair_price = env.step_price()
        bid, ask = agent.get_quotes(env.curr_time, fair_price)
        bid_filled, ask_filled = env.execute_orders(bid, ask)

        bid_edge_reward = agent.profit_incentive * (env.fair_price - bid) if bid_filled else 0
        ask_edge_reward = agent.profit_incentive * (ask - env.fair_price) if ask_filled else 0
        edge_reward = bid_edge_reward + ask_edge_reward

        curr_wealth = agent.update_state(fair_price, bid_filled, ask_filled, bid, ask)

        correction_reward = abs(prev_inventory) - abs(agent.inventory)
        correction_reward *= 10

        if step == steps - 1:
            terminal_penalty = agent.inventory**2
        else:
            terminal_penalty = 0

        reward = correction_reward + edge_reward - terminal_penalty

        rewards.append(reward)

        done = (step == steps - 1)
        dones.append(float(done))

        time_remaining = (config.T - env.curr_time) / config.T
        price_change = (env.fair_price - prev_price) / prev_price if prev_price > 0 else 0
        inventory_change = agent.inventory - prev_inventory

        obs = [agent.inventory, time_remaining, agent.current_spread, agent.current_skew,
               price_change, inventory_change]
        prev_inventory = agent.inventory

    return states, spread_actions, skew_actions, log_probs, rewards, values, dones

def ppo_update(model, optimizer, states, spread_actions, skew_actions, old_log_probs,
               advantages, returns, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5,
               n_epochs=4, batch_size=64):
    states = torch.tensor(states, dtype=torch.float32)
    spread_actions = torch.tensor(spread_actions, dtype=torch.long)
    skew_actions = torch.tensor(skew_actions, dtype=torch.long)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = len(states)
    total_loss = 0
    n_updates = 0

    for epoch in range(n_epochs):
        indices = torch.randperm(dataset_size)

        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_indices = indices[start:end]

            batch_states = states[batch_indices]
            batch_spread_actions = spread_actions[batch_indices]
            batch_skew_actions = skew_actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            new_log_probs, entropy, values = model.evaluate_actions(
                batch_states, batch_spread_actions, batch_skew_actions
            )
            values = values.squeeze()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, batch_returns)

            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            n_updates += 1

    return total_loss / n_updates if n_updates > 0 else 0

def train_ppo(model, env, agent, config, optimizer, n_episodes=500,
              gamma=0.99, lam=0.95, clip_epsilon=0.2, n_epochs=4, batch_size=64):
    """Train policy using PPO algorithm"""
    episode_rewards = []

    for episode in range(n_episodes):
        states, spread_actions, skew_actions, log_probs, rewards, values, dones = \
            run_episode(model, env, agent, config)

        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)

        loss = ppo_update(model, optimizer, states, spread_actions, skew_actions,
                          log_probs, advantages, returns, clip_epsilon,
                          n_epochs=n_epochs, batch_size=batch_size)

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}, Loss: {loss:.4f}")

    return episode_rewards

def run_eval_episode(model, env, agent, config):
    env.reset()
    agent.reset()
    steps = int(config.T / config.dt)
    prev_price = config.start_price
    prev_inventory = 0

    time_remaining = (config.T - env.curr_time) / config.T
    obs = [agent.inventory, time_remaining, agent.current_spread, agent.current_skew, 0, 0]

    for _ in range(steps):
        state = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            spread_logits, skew_logits, _ = model(state)
            spread_action = spread_logits.argmax().item()
            skew_action = skew_logits.argmax().item()

        agent.adjust_spread((spread_action, skew_action))

        prev_price = env.fair_price
        fair_price = env.step_price()
        bid, ask = agent.get_quotes(env.curr_time, fair_price)
        bid_filled, ask_filled = env.execute_orders(bid, ask)
        agent.update_state(fair_price, bid_filled, ask_filled, bid, ask)

        time_remaining = (config.T - env.curr_time) / config.T
        price_change = (env.fair_price - prev_price) / prev_price if prev_price > 0 else 0
        inventory_change = agent.inventory - prev_inventory

        obs = [agent.inventory, time_remaining, agent.current_spread, agent.current_skew,
               price_change, inventory_change]
        prev_inventory = agent.inventory

def evaluate_agent(model, env, agent, config, n_episodes=100):
    final_wealths = []
    for _ in range(n_episodes):
        run_eval_episode(model, env, agent, config)
        final_wealths.append(agent.wealth_history[-1])
    return np.mean(final_wealths), np.std(final_wealths)

def evaluate_standard(env, agent, config, n_episodes=100):
    final_wealths = []
    steps = int(config.T / config.dt)
    for _ in range(n_episodes):
        env.reset()
        agent.reset()
        for _ in range(steps):
            fair_price = env.step_price()
            bid, ask = agent.get_quotes(env.curr_time, fair_price)
            bid_filled, ask_filled = env.execute_orders(bid, ask)
            agent.update_state(fair_price, bid_filled, ask_filled, bid, ask)
        final_wealths.append(agent.wealth_history[-1])
    return np.mean(final_wealths), np.std(final_wealths)

def plot_results(episode_rewards, rl_mean, rl_std, naive_mean, naive_std, as_mean, as_std):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax1 = axes[0]
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, label=f'{window}-Episode Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('PPO Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    agents = ['Naive MM', 'Avellaneda-Stoikov', 'PPO Agent']
    means = [naive_mean, as_mean, rl_mean]
    stds = [naive_std, as_std, rl_std]
    colors = ['gray', 'orange', 'steelblue']
    bars = ax2.bar(agents, means, yerr=stds, capsize=5, color=colors)
    ax2.set_ylabel('Average Final Wealth')
    ax2.set_title('Performance Comparison (100 episodes)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, mean in zip(bars, means):
        va = 'bottom' if mean >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{mean:.2f}', ha='center', va=va)

    plt.tight_layout()
    plt.savefig('ppo_training_results.png', dpi=150)
    plt.show()

def plot_single_episode(model, env, rl_agent, as_agent, config):
    np.random.seed(42)
    env.reset()
    rl_agent.reset()

    rl_spreads, rl_skews, rl_inventories, rl_wealths = [], [], [], []
    steps = int(config.T / config.dt)
    prev_price = config.start_price
    prev_inventory = 0

    time_remaining = (config.T - env.curr_time) / config.T
    obs = [rl_agent.inventory, time_remaining, rl_agent.current_spread, rl_agent.current_skew, 0, 0]

    for step in range(steps):
        state = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            spread_logits, skew_logits, _ = model(state)
            spread_action = spread_logits.argmax().item()
            skew_action = skew_logits.argmax().item()

        rl_agent.adjust_spread((spread_action, skew_action))
        rl_spreads.append(rl_agent.current_spread)
        rl_skews.append(rl_agent.current_skew)

        prev_price = env.fair_price
        fair_price = env.step_price()
        bid, ask = rl_agent.get_quotes(env.curr_time, fair_price)
        bid_filled, ask_filled = env.execute_orders(bid, ask)
        rl_agent.update_state(fair_price, bid_filled, ask_filled, bid, ask)

        rl_inventories.append(rl_agent.inventory)
        rl_wealths.append(rl_agent.wealth_history[-1])

        time_remaining = (config.T - env.curr_time) / config.T
        price_change = (env.fair_price - prev_price) / prev_price if prev_price > 0 else 0
        inventory_change = rl_agent.inventory - prev_inventory
        obs = [rl_agent.inventory, time_remaining, rl_agent.current_spread, rl_agent.current_skew,
               price_change, inventory_change]
        prev_inventory = rl_agent.inventory

    rl_price_history = env.price_history.copy()
    rl_time_history = env.time_history.copy()

    np.random.seed(42)
    env.reset()
    as_agent.reset()

    as_spreads, as_skews, as_inventories, as_wealths = [], [], [], []

    for step in range(steps):
        fair_price = env.step_price()
        bid, ask = as_agent.get_quotes(env.curr_time, fair_price)

        as_skew = as_agent.reservation_price(fair_price, env.curr_time) - fair_price
        as_spread = as_agent.optimal_spread(env.curr_time)

        bid_filled, ask_filled = env.execute_orders(bid, ask)
        as_agent.update_state(fair_price, bid_filled, ask_filled, bid, ask)

        as_spreads.append(as_spread)
        as_skews.append(as_skew)
        as_inventories.append(as_agent.inventory)
        as_wealths.append(as_agent.wealth_history[-1])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    axes[0, 0].plot(rl_time_history, rl_price_history)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Fair Price')
    axes[0, 0].set_title('Price Evolution')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(rl_inventories, rl_skews, alpha=0.5, s=10, label='PPO Agent', color='steelblue')
    axes[0, 1].scatter(as_inventories, as_skews, alpha=0.5, s=10, label='Avellaneda-Stoikov', color='orange')
    axes[0, 1].set_xlabel('Inventory')
    axes[0, 1].set_ylabel('Skew')
    axes[0, 1].set_title('Inventory vs Skew (should be negatively correlated)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(range(steps), rl_inventories, label='PPO Agent', color='steelblue')
    axes[1, 0].plot(range(steps), as_inventories, label='Avellaneda-Stoikov', color='orange')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Inventory')
    axes[1, 0].set_title('Inventory Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(range(steps), rl_skews, label='PPO Agent', color='steelblue')
    axes[1, 1].plot(range(steps), as_skews, label='Avellaneda-Stoikov', color='orange')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Skew')
    axes[1, 1].set_title('Quote Skew Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(range(steps), rl_wealths, label='PPO Agent', color='steelblue')
    axes[2, 0].plot(range(steps), as_wealths, label='Avellaneda-Stoikov', color='orange')
    axes[2, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Wealth')
    axes[2, 0].set_title('Cumulative PnL Comparison')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].scatter(rl_inventories, rl_skews, alpha=0.5, s=10, label='PPO Agent', color='steelblue')
    axes[2, 1].scatter(as_inventories, as_skews, alpha=0.5, s=10, label='Avellaneda-Stoikov', color='orange')
    axes[2, 1].set_xlabel('Inventory')
    axes[2, 1].set_ylabel('Skew')
    axes[2, 1].set_title('Inventory vs Skew (should be negatively correlated)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ppo_episode_detail.png', dpi=150)
    plt.show()

def main():
    config = MarketConfig(T=1.0, dt=0.005, sigma=0.5, k=1.5, A=140)
    env = MarketEnv(config)

    model = ActorCritic()
    agent = RLMarketMaker(spread_delta=0.1, skew_delta=0.1,
                          inventory_risk_aversion=0.001, profit_incentive=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)

    n_episodes = 500

    print("Training PPO agent...")

    episode_rewards = train_ppo(
        model, env, agent, config, optimizer,
        n_episodes=n_episodes, gamma=0.95, lam=0.95,
        clip_epsilon=0.2, n_epochs=4, batch_size=128
    )

    print("Training complete!")

    print("Evaluating PPO agent...")
    rl_mean, rl_std = evaluate_agent(model, env, agent, config, n_episodes=100)
    print(f"PPO Agent - Mean: {rl_mean:.2f}, Std: {rl_std:.2f}")

    print("Evaluating naive baseline...")
    naive_agent = NaiveMarketMaker(spread=0.5)
    naive_mean, naive_std = evaluate_standard(env, naive_agent, config, n_episodes=100)
    print(f"Naive MM - Mean: {naive_mean:.2f}, Std: {naive_std:.2f}")

    print("Evaluating Avellaneda-Stoikov agent...")
    as_agent = AvellanedaStoikov(T=config.T, sigma=config.sigma, gamma=1, k=config.k)
    as_mean, as_std = evaluate_standard(env, as_agent, config, n_episodes=100)
    print(f"Avellaneda-Stoikov - Mean: {as_mean:.2f}, Std: {as_std:.2f}")

    print("Generating plots...")
    plot_results(episode_rewards, rl_mean, rl_std, naive_mean, naive_std, as_mean, as_std)
    plot_single_episode(model, env, agent, as_agent, config)

if __name__ == "__main__":
    main()
