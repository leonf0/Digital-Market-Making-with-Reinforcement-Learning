from src.MarketEnv import MarketEnv and MarketConfig
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque

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
