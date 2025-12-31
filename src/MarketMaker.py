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
