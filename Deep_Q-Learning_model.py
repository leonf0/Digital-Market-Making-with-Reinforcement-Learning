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

class DQN(nn.Module):  
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),  
            nn.Linear(64, 32),
            nn.ReLU(),  
            nn.Linear(32, 9)  
        )

    def forward(self, state):
        return self.net(state)

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

def action_to_index(action):
    spread_action, skew_action = action
    return spread_action * 3 + skew_action

def index_to_action(index):
    return (index // 3, index % 3)

def choose_dqn_action(model, obs, epsilon):
    if torch.rand(()).item() < epsilon:
        spread_action = torch.randint(3, size=()).item()  
        skew_action = torch.randint(3, size=()).item()    
        return (spread_action, skew_action)
    else:
        state = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            Q_values = model(state)
        action_index = Q_values.argmax().item()  
        return index_to_action(action_index)

def sample_experiences(replay_buffer, batch_size):  
    indices = torch.randint(len(replay_buffer), size=(batch_size,))  
    batch = [replay_buffer[index] for index in indices.tolist()]
    return [to_tensor([exp[i] for exp in batch]) for i in range(4)]  

def to_tensor(data):
    array = np.stack(data)
    dtype = torch.float32 if array.dtype == np.float64 else None
    return torch.as_tensor(array, dtype=dtype)

def compute_returns(rewards, discount_factor):
    returns = rewards[:]
    for i in range(len(returns) - 2, -1, -1):
        returns[i] += returns[i + 1] * discount_factor
    return torch.tensor(returns, dtype=torch.float32)

def run_and_record_episode(model, env, replay_buffer, epsilon, agent, config):
    env.reset()
    agent.reset()
    rewards = []
    steps = int(config.T / config.dt)
    prev_wealth = 0
    prev_price = config.start_price
    prev_inventory = 0

    time_remaining = (config.T - env.curr_time) / config.T
    price_change = 0
    obs = [
        agent.inventory,
        time_remaining,
        agent.current_spread,
        agent.current_skew,  
        price_change,
        0  
    ]

    for _ in range(steps):
        action = choose_dqn_action(model, obs, epsilon)
        agent.adjust_spread(action)

        prev_price = env.fair_price
        fair_price = env.step_price()
        bid, ask = agent.get_quotes(env.curr_time, fair_price)
        bid_filled, ask_filled = env.execute_orders(bid, ask)

        bid_edge_reward = agent.profit_incentive * (env.fair_price - bid) if bid_filled else 0
        ask_edge_reward = agent.profit_incentive * (ask - env.fair_price) if ask_filled else 0

        curr_wealth = agent.update_state(fair_price, bid_filled, ask_filled, bid, ask)
        inventory_penalty = agent.inventory_risk_aversion * ((agent.inventory * env.fair_price)**2)
        reward = bid_edge_reward + ask_edge_reward - inventory_penalty
        rewards.append(reward)

        time_remaining = (config.T - env.curr_time) / config.T
        price_change = (env.fair_price - prev_price) / prev_price if prev_price > 0 else 0
        inventory_change = agent.inventory - prev_inventory

        next_obs = [
            agent.inventory,
            time_remaining,
            agent.current_spread,
            agent.current_skew,
            price_change,
            inventory_change
        ]

        action_index = action_to_index(action)
        experience = (obs, action_index, reward, next_obs)  
        replay_buffer.append(experience)

        obs = next_obs
        prev_wealth = curr_wealth
        prev_inventory = agent.inventory

    return rewards

def dqn_training_step(model, optimizer, criterion, replay_buffer, batch_size,
                      discount_factor=0.95): 
    experiences = sample_experiences(replay_buffer, batch_size)
    state, action, reward, next_state = experiences

    with torch.inference_mode():
        next_Q_values = model(next_state) 
    max_next_Q_value, _ = next_Q_values.max(dim=1)
    target_Q_values = reward + discount_factor * max_next_Q_value

    all_Q_values = model(state)
    action = action.long()  
    Q_values = all_Q_values.gather(dim=1, index=action.unsqueeze(1))

    loss = criterion(Q_values, target_Q_values.unsqueeze(1))  

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_dqn(model, env, agent, config, replay_buffer, optimizer, criterion,
              n_episodes=500, warmup=30, batch_size=32, discount_factor=0.95):
    episode_rewards = []

    for episode in range(n_episodes):
        epsilon = max(1 - episode / n_episodes, 0.01)
        rewards = run_and_record_episode(model, env, replay_buffer, epsilon, agent, config)
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        if episode >= warmup and len(replay_buffer) >= batch_size:
            dqn_training_step(model, optimizer, criterion, replay_buffer, batch_size, discount_factor)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    return episode_rewards

def run_episode(model, env, agent, config):
    env.reset()
    agent.reset()
    steps = int(config.T / config.dt)
    prev_price = config.start_price
    prev_inventory = 0

    time_remaining = (config.T - env.curr_time) / config.T
    obs = [agent.inventory, time_remaining, agent.current_spread, agent.current_skew, 0, 0]

    for _ in range(steps):
        action = choose_dqn_action(model, obs, epsilon=0.0)  # greedy for evaluation
        agent.adjust_spread(action)

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
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, label=f'{window}-Episode Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('DQN Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    agents = ['Naive MM', 'Avellaneda-Stoikov', 'DQN Agent']
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
    plt.savefig('dqn_training_results.png', dpi=150)
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
        action = choose_dqn_action(model, obs, epsilon=0.0)
        rl_agent.adjust_spread(action)
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

    axes[0, 1].scatter(rl_inventories, rl_skews, alpha=0.5, s=10, label='DQN Agent', color='steelblue')
    axes[0, 1].scatter(as_inventories, as_skews, alpha=0.5, s=10, label='Avellaneda-Stoikov', color='orange')
    axes[0, 1].set_xlabel('Inventory')
    axes[0, 1].set_ylabel('Skew')
    axes[0, 1].set_title('Inventory vs Skew (should be negatively correlated)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(range(steps), rl_inventories, label='DQN Agent', color='steelblue')
    axes[1, 0].plot(range(steps), as_inventories, label='Avellaneda-Stoikov', color='orange')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Inventory')
    axes[1, 0].set_title('Inventory Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(range(steps), rl_skews, label='DQN Agent', color='steelblue')
    axes[1, 1].plot(range(steps), as_skews, label='Avellaneda-Stoikov', color='orange')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Skew')
    axes[1, 1].set_title('Quote Skew Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(range(steps), rl_wealths, label='DQN Agent', color='steelblue')
    axes[2, 0].plot(range(steps), as_wealths, label='Avellaneda-Stoikov', color='orange')
    axes[2, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Wealth')
    axes[2, 0].set_title('Cumulative PnL Comparison')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_episode_detail.png', dpi=150)
    plt.show()

def main():
    config = MarketConfig(T=1.0, dt=0.005, sigma=0.5, k=1.5, A=140)
    env = MarketEnv(config)

    model = DQN()
    agent = RLMarketMaker(spread_delta=0.1, skew_delta=0.1,
                          inventory_risk_aversion=1, profit_incentive=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=100_000)

    n_episodes = 20000
    warmup = 256
    batch_size = 128

    print("Training DQN agent...")

    episode_rewards = train_dqn(
        model, env, agent, config, replay_buffer, optimizer, criterion,
        n_episodes=n_episodes, warmup=warmup, batch_size=batch_size
    )

    print("Training complete!")

    print("Evaluating DQN agent...")
    rl_mean, rl_std = evaluate_agent(model, env, agent, config, n_episodes=100)
    print(f"DQN Agent - Mean: {rl_mean:.2f}, Std: {rl_std:.2f}")

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
