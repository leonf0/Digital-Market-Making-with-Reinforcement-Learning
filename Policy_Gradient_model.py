import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass


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


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, state):
        return self.net(state)


class RLMarketMaker(MarketMaker):
    def __init__(self, spread_delta=0.1, skew_delta=0.1, inventory_risk_aversion=1.0, profit_incentive=1000):
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


def choose_action(model, obs):
    state = torch.as_tensor(obs, dtype=torch.float32)
    logits = model(state)

    spread_dist = torch.distributions.Categorical(logits=logits[:3])
    skew_dist = torch.distributions.Categorical(logits=logits[3:])

    spread_action = spread_dist.sample()
    skew_action = skew_dist.sample()
    log_prob = spread_dist.log_prob(spread_action) + skew_dist.log_prob(skew_action)

    return (spread_action.item(), skew_action.item()), log_prob


def compute_returns(rewards, discount_factor):
    returns = rewards[:]
    for i in range(len(returns) - 2, -1, -1):
        returns[i] += returns[i + 1] * discount_factor
    return torch.tensor(returns, dtype=torch.float32)


def run_episode(model, env, agent, config):
    env.reset()
    agent.reset()

    log_probs = []
    rewards = []
    steps = int(config.T / config.dt)
    prev_wealth = 0
    prev_price = config.start_price
    prev_inventory = 0

    for _ in range(steps):
        time_remaining = (config.T - env.curr_time) / config.T
        price_change = (env.fair_price - prev_price) / prev_price if prev_price > 0 else 0

        obs = [
            agent.inventory,
            time_remaining,
            agent.current_spread,
            agent.current_skew,
            price_change,
            (agent.inventory - prev_inventory)
        ]

        action, log_prob = choose_action(model, obs)
        log_probs.append(log_prob)
        agent.adjust_spread(action)

        prev_price = env.fair_price
        fair_price = env.step_price()
        bid, ask = agent.get_quotes(env.curr_time, fair_price)
        bid_filled, ask_filled = env.execute_orders(bid, ask)

        bid_edge_reward = agent.profit_incentive*(env.fair_price - bid) if bid_filled else 0
        ask_edge_reward = agent.profit_incentive*(ask - env.fair_price) if ask_filled else 0

        curr_wealth = agent.update_state(fair_price, bid_filled, ask_filled, bid, ask)

        inventory_penalty = agent.inventory_risk_aversion * ((agent.inventory*env.fair_price)**2)*config.dt
        rewards.append(bid_edge_reward + ask_edge_reward - inventory_penalty)

        prev_wealth = curr_wealth
        prev_inventory = agent.inventory

    return log_probs, rewards


def train_reinforce(model, optimizer, env, agent, config, n_episodes, discount_factor=0.99):
    episode_rewards = []
    baseline = 0.0

    for episode in range(n_episodes):
        log_probs, rewards = run_episode(model, env, agent, config)
        returns = compute_returns(rewards, discount_factor)

        baseline = 0.9 * baseline + 0.1 * returns.mean().item()
        advantages = returns - baseline

        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        loss = sum(-lp * adv for lp, adv in zip(log_probs, advantages))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg:.2f}")

    return episode_rewards


def evaluate_agent(model, env, agent, config, n_episodes=100):
    final_wealths = []
    for _ in range(n_episodes):
        run_episode(model, env, agent, config)
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
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(episode_rewards)), moving_avg, label=f'{window}-Episode Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    agents = ['Naive MM', 'Avellaneda-Stoikov', 'RL Agent']
    means = [naive_mean, as_mean, rl_mean]
    stds = [naive_std, as_std, rl_std]
    colors = ['gray', 'orange', 'steelblue']
    bars = ax2.bar(agents, means, yerr=stds, capsize=5, color=colors)
    ax2.set_ylabel('Average Final Wealth')
    ax2.set_title('Performance Comparison (100 episodes)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, mean in zip(bars, means):
        va = 'bottom' if mean >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{mean:.2f}', ha='center', va=va)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()


def plot_single_episode(model, env, rl_agent, as_agent, config):
    np.random.seed(42)
    env.reset()
    rl_agent.reset()

    steps = int(config.T / config.dt)
    prev_price = config.start_price

    for _ in range(steps):
        time_remaining = (config.T - env.curr_time) / config.T
        price_change = (env.fair_price - prev_price) / prev_price if prev_price > 0 else 0
        obs = [rl_agent.inventory, time_remaining, rl_agent.current_spread,
               rl_agent.current_skew, price_change, 0]

        state = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits = model(state)
            action = (logits[:3].argmax().item(), logits[3:].argmax().item())

        rl_agent.adjust_spread(action)
        prev_price = env.fair_price
        fair_price = env.step_price()
        bid, ask = rl_agent.get_quotes(env.curr_time, fair_price)
        bid_filled, ask_filled = env.execute_orders(bid, ask)
        rl_agent.update_state(fair_price, bid_filled, ask_filled, bid, ask)

    rl_price_history = env.price_history.copy()
    rl_time_history = env.time_history.copy()

    np.random.seed(1)
    env.reset()
    as_agent.reset()

    for _ in range(steps):
        fair_price = env.step_price()
        bid, ask = as_agent.get_quotes(env.curr_time, fair_price)
        bid_filled, ask_filled = env.execute_orders(bid, ask)
        as_agent.update_state(fair_price, bid_filled, ask_filled, bid, ask)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(rl_time_history, rl_price_history)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Fair Price')
    axes[0, 0].set_title('Price Evolution')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(rl_agent.inventory_history, label='RL Agent', color='steelblue')
    axes[0, 1].plot(as_agent.inventory_history, label='Avellaneda-Stoikov', color='orange')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Inventory')
    axes[0, 1].set_title('Inventory Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(rl_agent.wealth_history, label='RL Agent', color='steelblue')
    axes[1, 0].plot(as_agent.wealth_history, label='Avellaneda-Stoikov', color='orange')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Wealth')
    axes[1, 0].set_title('Cumulative PnL Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis('off')
    summary = f"Final Results:\nRL Agent: ${rl_agent.wealth_history[-1]:.2f}\nA-S Agent: ${as_agent.wealth_history[-1]:.2f}"
    axes[1, 1].text(0.5, 0.5, summary, ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('episode_detail.png', dpi=150)
    plt.show()


def main():
    config = MarketConfig()
    env = MarketEnv(config)

    model = PolicyNetwork()
    rl_agent = RLMarketMaker(spread_delta=0.1, skew_delta=0.1,inventory_risk_aversion=1.0, profit_incentive=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training Policy Network RL agent...")

    episode_rewards = train_reinforce(model, optimizer, env, rl_agent, config, n_episodes=500)

    print("Training complete!")

    print("Evaluating agents...")
    rl_mean, rl_std = evaluate_agent(model, env, rl_agent, config)
    print(f"RL Agent        - Mean: {rl_mean:.2f}, Std: {rl_std:.2f}")

    naive_agent = NaiveMarketMaker(spread=0.5)
    naive_mean, naive_std = evaluate_standard(env, naive_agent, config)
    print(f"Naive MM        - Mean: {naive_mean:.2f}, Std: {naive_std:.2f}")

    as_agent = AvellanedaStoikov(T=config.T, sigma=config.sigma, gamma=1.0, k=config.k)
    as_mean, as_std = evaluate_standard(env, as_agent, config)
    print(f"Avellaneda-Stoikov - Mean: {as_mean:.2f}, Std: {as_std:.2f}")

    plot_results(episode_rewards, rl_mean, rl_std, naive_mean, naive_std, as_mean, as_std)
    plot_single_episode(model, env, rl_agent, as_agent, config)


if __name__ == "__main__":
    main()
