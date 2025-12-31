import numpy as np
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
