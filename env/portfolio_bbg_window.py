import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from gymnasium.utils import seeding
import collections
import math
from gym.spaces import Dict, Box
import os
import datetime


#! ERRORS: sharpe and loss weight need to be referenced with self.


class Portfolio_BBG_window_v1(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df,
                 action_space,
                 stock_dim,
                 stock_list,
                 short_selling_allowed,
                 num_stock_shares: list[int],
                 take_leverage_allowed,
                 trade_cost_pct,
                 initial_amount: int,
                 state_space,
                 indicators,
                 reward_scaling,
                 previous_state=[],
                 day=0,
                 initial=True,
                 max_drawdown_threshold: float = 0.1,
                 print_verbosity=1,
                 lookback_window=50,
                 hmax: int = 100,
                 drawdown_penalty_Factor: float = 0.1,
                 drawdown_penalty_weight: float = 0.1,
                 position_flipping_window: int = 5,
                 position_flipping_weight: float = 0.1,
                 position_flipping_factor: float = 0.1,
                 daily_return_weight: float = 0.1,
                 ):

        self.drawdown_penalty_Factor = drawdown_penalty_Factor
        self.drawdown_penalty_weight = drawdown_penalty_weight
        self.position_flipping_window = position_flipping_window
        self.position_flipping_weight = position_flipping_weight
        self.position_flipping_factor = position_flipping_factor
        self.daily_return_weight = daily_return_weight

        self.terminal = False
        self.max_drawdown_threshold = max_drawdown_threshold
        self.lookback_window = lookback_window
        self.action_space = action_space
        self.previous_state = previous_state
        self.stock_dim = stock_dim
        self.reward_scaling = reward_scaling

        self.stock_list = stock_list
        self.num_stock_shares = num_stock_shares
        self.short_selling_allowed = short_selling_allowed
        self.take_leverage_allowed = take_leverage_allowed
        self.day = day + self.lookback_window
        self.stock_holdings_memory = [
            np.zeros(self.stock_dim) for _ in range(self.lookback_window)]

        self.df = df
        self.hmax = hmax
        self.trade_cost_pct = trade_cost_pct
        self.state_space = state_space
        self.indicator_list = indicators
        self.initial = initial
        self.initial_amount = initial_amount
        self.data = self.df.loc[self.day -
                                self.lookback_window: self.day - 1, :]

        self.print_verbosity = print_verbosity
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_space,))
        self.expected_length = 1 + 2 * self.lookback_window + \
            self.lookback_window * len(self.indicator_list)
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(self.expected_length,))
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(len(self._update_state()),))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(
            self.lookback_window * (1 + self.stock_dim * 2 + len(self.indicator_list)),))

        self.state = self._initiate_state()

        # Initiate Reward
        self.cost = 0
        self.trades = 0
        self.reward = 0
        self.episode = 0
        self.current_step_cost = 0
        # memorize total balance
        self.asset_memory = [self.initial_amount + np.sum(
            np.array(self.num_stock_shares) *
            np.array(self.stock_prices)
        )]
        self.rewards_memory = []

        self.portfolio_return_memory = [0]
        self.actions_memory = []
        self.state_memory = ([])
        self.date_memory = [self._get_date()]

        self.portfolio_memory = {"cash": [self.initial_amount]}
        for stock, shares in zip(self.stock_list, self.num_stock_shares):
            self.portfolio_memory[stock] = [shares]

    @property
    def cash_balance(self):
        # Assuming cash balance is the same for all 5 days, return the last day's balance.
        # If you have varying balances for each day, you may need to adjust this.
        return self.state[-1, 0]

    @cash_balance.setter
    def cash_balance(self, value):
        self.state[-1, 0] = value

    @property
    def stock_prices(self):
        # This will return a (5, stock_dim) array.
        return self.state[:, 1:1+self.stock_dim]

    @stock_prices.setter
    def stock_prices(self, values):
        self.state[-1, 1:1+self.stock_dim] = values

    @property
    def stock_shares(self):
        # This will return a (5, stock_dim) array.
        return self.state[:, 1+self.stock_dim:1+2*self.stock_dim]

    @property
    def indicators(self):
        # This will return a (5, indicators_dim) array.
        return self.state[:, 1+2*self.stock_dim:]

    def _sell_stock(self, index, action):
        if self.short_selling_allowed:
            print("Short sell teriggere")
            sell_num_shares = math.floor(abs(action))
        else:
            sell_num_shares = math.floor(
                min(self.stock_shares[-1][index], abs(action)))
        sell_amount = self.stock_prices[-1][index] * \
            sell_num_shares * (1 - self.trade_cost_pct)
        self.stock_shares[-1][index] -= sell_num_shares

        self.cost += (self.stock_prices[-1][index]
                      * sell_num_shares * self.trade_cost_pct)
        self.trades += 1
        self.cash_balance += sell_amount
        self.current_step_cost += self.stock_prices[index] * \
            sell_num_shares * self.trade_cost_pct

        return sell_num_shares

    def _buy_stock(self, index, action):
        if self.take_leverage_allowed:
            buy_num_shares = math.floor(action)
        else:
            available_amount = self.cash_balance / \
                (self.stock_prices[-1][index] * (1 + self.trade_cost_pct))
            buy_num_shares = math.floor(min(available_amount, action))
        buy_amount = self.stock_prices[-1][index] * \
            buy_num_shares * (1 + self.trade_cost_pct)

        self.stock_shares[-1][index] += buy_num_shares

        self.cost += (self.stock_prices[-1][index]
                      * buy_num_shares * self.trade_cost_pct)
        self.trades += 1
        self.cash_balance -= buy_amount
        self.current_step_cost += self.stock_prices[index] * \
            buy_num_shares * self.trade_cost_pct

        return buy_num_shares

    def calculate_reward(self, begin_total_asset, end_total_asset):
        annualized_return = (
            ((end_total_asset / self.asset_memory[0]) ** (365 / (self.day - self.lookback_window)))-1)*100
        total_asset_reward = end_total_asset - \
            begin_total_asset - self.current_step_cost
        daily_return = (end_total_asset - begin_total_asset -
                        self.current_step_cost) / begin_total_asset
        drawdown_penalty = 0
        if self.asset_memory[-1] < (1 - self.max_drawdown_threshold) * max(self.asset_memory):
            drawdown_penalty = -self.drawdown_penalty_Factor * self.drawdown_penalty_weight
        frequent_flipping_pentalty = 0
        if len(self.actions_memory) > self.position_flipping_window:
            # Rapid flipping between buying and selling
            if sum(self.actions_memory[-self.position_flipping_window:]) == 0:
                frequent_flipping_pentalty -= self.position_flipping_factor * \
                    self.position_flipping_weight
        return annualized_return + drawdown_penalty + frequent_flipping_pentalty + daily_return
        return total_asset_reward * self.reward_scaling + annualized_return * self.reward_scaling

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            self._handle_terminal_condition()
        else:
            self.current_step_cost = 0
            actions = actions * self.hmax
            actions = actions.astype(int)
            begin_total_asset = self.cash_balance + \
                np.sum(self.stock_prices[-1] * self.stock_shares[-1])

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-
                                        1][: np.where(actions > 0)[0].shape[0]]
            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            self.day += 1

            self.data = self.df.loc[self.day -
                                    self.lookback_window: self.day - 1, :]
            new_prices = self.data['close'].values
            self.stock_prices = new_prices[-1]

            self.stock_holdings_memory.append(self.stock_shares[-1].copy())
            if len(self.stock_holdings_memory) > self.lookback_window:
                self.stock_holdings_memory.pop(0)
            end_total_asset = self.cash_balance + \
                np.sum(self.stock_prices[-1] * self.stock_shares[-1])
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            portfolio_return = (end_total_asset / begin_total_asset) - 1
            self.portfolio_return_memory.append(portfolio_return)
            self.reward = self.calculate_reward(
                begin_total_asset, end_total_asset)
            self.rewards_memory.append(float(self.reward))

            self.state_memory.append(self.state)
            self.state = self._update_state()

            self.portfolio_memory["cash"].append(self.cash_balance)
            for i, stock_name in enumerate(self.stock_list):
                self.portfolio_memory[stock_name].append(
                    self.stock_shares[-1, i])
        return self.state.flatten(), self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        self.day = self.lookback_window
        self.data = self.df.loc[self.day -
                                self.lookback_window: self.day - 1, :]

        self.cash_balance = self.initial_amount

        self.state = self._initiate_state()
        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(self.stock_prices * self.stock_shares)
            ]
        else:
            print("self.initial False not implemented yet")
            previous_total_asset = self.cash_balance + \
                np.sum(self.stock_prices * self.stock_shares)
            self.asset_memory = [previous_total_asset]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.portfolio_return_memory = [0]
        self.stock_holdings_memory = [
            np.zeros(self.stock_dim) for _ in range(self.lookback_window)]

        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        self.portfolio_memory = {"cash": [self.initial_amount]}
        for stock in self.stock_list:
            self.portfolio_memory[stock] = [
                self.num_stock_shares[self.stock_list.index(stock)]]

        return self.state.flatten(), {}

    def _initiate_state(self):
        if self.initial:
            # Initialize cash_balance as a single value repeated over lookback_window days.
            cash_balance = np.full(
                (self.lookback_window, 1), self.initial_amount)

            # Convert stock prices and indicators to a 2D array with shape (lookback_window, feature_dim).
            stock_prices = np.array(self.data.close.values).reshape(
                self.lookback_window, self.stock_dim)
            indicators = np.array(self.data[self.indicator_list].values).reshape(
                self.lookback_window, len(self.indicator_list))

            # Initialize stock shares as zeros.
            stock_shares = np.zeros((self.lookback_window, self.stock_dim))

            # Concatenate all arrays horizontally to form the state.
            state = np.hstack(
                (cash_balance, stock_prices, stock_shares, indicators))

        else:
            raise ValueError(
                "Not implemented yet - triggered because self.initial was False")

        return state

    def _update_state(self):
        # Ensure cash_balance is a 2D array with shape (lookback_window, 1).
        cash_balance = np.full((self.lookback_window, 1), self.cash_balance)

        # Ensure stock_prices is a 2D array.
        stock_prices = self.stock_prices.reshape(
            self.lookback_window, self.stock_dim)

        # Convert stock_shares and indicators to 2D arrays.
        stock_shares = np.array(
            self.stock_holdings_memory[-self.lookback_window:]).reshape(self.lookback_window, self.stock_dim)

        indicators = self.data[self.indicator_list].values

        # Stack these arrays vertically.
        return np.hstack([cash_balance, stock_prices, stock_shares, indicators])

    # ------------LESS RELEVANT-----------------

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[-1]
        else:
            date = self.data.date.iloc[-1]
        return date

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def calculate_sharpe_ratio(self):
        returns = np.array(self.portfolio_return_memory)
        std_returns = np.std(returns)
        if std_returns == 0:
            std_returns = 1e-6
        return np.mean(returns) / std_returns * np.sqrt(252)

    def _handle_terminal_condition(self):
        end_total_asset = self.cash_balance + \
            np.sum(self.stock_prices[-1] * self.stock_shares[-1])
        df_total_value = pd.DataFrame({'account_value': self.asset_memory})
        total_reward = end_total_asset - self.initial_amount
        df_total_value["date"] = self.date_memory
        df_total_value['daily_return'] = df_total_value['account_value'].pct_change(
            1)
        sharpe = 0
        if df_total_value['daily_return'].std() != 0:
            sharpe = (
                (252**0.5) * df_total_value['daily_return'].mean() /
                df_total_value['daily_return'].std()
            )
        df_rewards = pd.DataFrame(self.rewards_memory)
        df_rewards.columns = ['account_rewards']
        df_rewards['date'] = self.date_memory[:-1]
        df_daily_return = pd.DataFrame(self.portfolio_return_memory)
        df_daily_return.columns = ['daily_return']
        SPX_ann_return = (((self.df.close.iloc[-1] / self.df.close.iloc[self.lookback_window]) ** (
            365 / (self.day - self.lookback_window))) - 1) * 100

        annualized_return = (
            ((end_total_asset / self.asset_memory[0]) ** (365 / (self.day - self.lookback_window)))-1)*100
        if self.episode % self.print_verbosity == 0:
            self.print_run_outcome(
                end_total_asset, total_reward, df_total_value, sharpe, annualized_return, SPX_ann_return)
        self.save_and_print_dfs()
        return self.state.flatten(), self.reward, self.terminal, False, {}

    def print_run_outcome(self, end_total_asset, total_reward, df_total_value, sharpe, annualized_return, SPX_ann_return):
        print(f"day: {self.day}, episode: {self.episode}")
        print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
        print(f"end_total_asset: {end_total_asset:0.2f}")
        print(f"total_reward: {total_reward:0.2f}")
        print("self.cost:", self.cost)
        # print(f"total_cost: {self.cost[0]:0.2f}")
        print(f"total_trades: {self.trades}")
        if df_total_value["daily_return"].std() != 0:
            print(f"Sharpe: {sharpe:0.3f}")
        print(f"Annual Return: {annualized_return:0.2f}%")
        print(f"SPX Annual Return: {SPX_ann_return:0.2f}%")
        print("=================================")

    def memory_collection(self):
        date_list = self.date_memory[:-1]
        asset_list = self.asset_memory[:-1]
        stock_holdings_list = self.stock_holdings_memory[:-1]
        return_list = self.portfolio_return_memory[:-1]
        action_list = self.actions_memory[:-1]
        portfolio_memory = self.portfolio_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        df_stock_holdings = pd.DataFrame(
            {"date": date_list, "stock_holdings": stock_holdings_list})
        df_return_memory = pd.DataFrame(
            {"date": date_list, "daily_return": return_list})
        df_actions = pd.DataFrame(
            {"date": date_list, "actions": action_list})
        df_portfolio_memory = pd.DataFrame(date_list, columns=["date"])
        for asset, values in portfolio_memory.items():
            df_portfolio_memory[asset] = values[:-1]

        return df_account_value, df_stock_holdings, df_return_memory, df_actions, df_portfolio_memory

    def save_and_print_dfs(self):
        date_list = self.date_memory[:-1]
        asset_list = self.asset_memory[:-1]

        # stock_holdings_list = self.stock_holdings_memory[self.lookback_window-1:]
        # stock_holdings_list = self.stock_holdings_memory[:-1]
        # stock_holdings_list = self.stock_holdings_memory[self.lookback_window:]

        return_list = self.portfolio_return_memory[:-1]

        action_list = self.actions_memory[:-1]

        portfolio_memory = self.portfolio_memory

        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )

        # df_stock_holdings = pd.DataFrame(
        #     {"date": date_list, "stock_holdings": stock_holdings_list})

        df_return_memory = pd.DataFrame(
            {"date": date_list, "daily_return": return_list})

        df_actions = pd.DataFrame(
            {"date": date_list[:-1], "actions": action_list})

        df_portfolio_memory = pd.DataFrame(date_list, columns=["date"])

        for asset, values in portfolio_memory.items():
            df_portfolio_memory[asset] = values[:-1]

        # Combine DataFrames
        # combined_df = df_account_value.merge(df_stock_holdings, on="date")
        combined_df = df_account_value.merge(df_return_memory, on="date")
        combined_df = combined_df.merge(df_actions, on="date")
        combined_df = combined_df.merge(df_portfolio_memory, on="date")

        # Print combined DataFrame

        # Save to CSV with a unique timestamped filename
        save_path = "logs"  # specify the desired directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"bbg_env_runs_{current_time}.csv"
        combined_df.to_csv(os.path.join(save_path, filename), index=False)

    # ? ist combined
    # def save_asset_memory(self):

    #     date_list = self.date_memory[:-1]
    #     asset_list = self.asset_memory[:-1]
    #     # print(len(date_list))
    #     # print(len(asset_list))
    #     print(len(date_list))
    #     print(len(asset_list))
    #     df_account_value = pd.DataFrame(
    #         {"date": date_list, "account_value": asset_list}
    #     )
    #     return df_account_value
     # ? ist combined
    # def save_stock_holdings_memory(self):
    #     date_list = self.date_memory[:-1]
    #     stock_holdings_list = self.stock_holdings_memory[:-1]
    #     print(len(date_list))
    #     print(len(stock_holdings_list))
    #     df_stock_holdinds = pd.DataFrame(
    #         {"date": date_list, "stock_holdings": stock_holdings_list})
    #     return df_stock_holdinds
    # ? ist combined
    # def save_return_memory(self):
    #     date_list = self.date_memory[:-1]
    #     return_list = self.portfolio_return_memory[:-1]
    #     print(len(date_list))
    #     print(len(return_list))
    #     df_return_memory = pd.DataFrame(
    #         {"date": date_list, "daily_return": return_list})
    #     return df_return_memory
    # ? ist combined
    # def save_action_memory(self):
    #     if len(self.df.tic.unique()) > 1:
    #         raise ValueError("Stocks are more than one")
    #     else:
    #         date_list = self.date_memory[:-1]
    #         action_list = self.actions_memory
    #         print(len(date_list))
    #         print(len(action_list))
    #         df_actions = pd.DataFrame(
    #             {"date": date_list, "actions": action_list})
    #     return df_actions
     # ? ist combined
    # def save_portfolio_memory(self):
    #     portfolio_memory = self.portfolio_memory
    #     date_list = self.date_memory[:-1]

    #     # Create a DataFrame with dates
    #     df_portfolio_memory = pd.DataFrame(date_list, columns=["date"])

    #     # Add columns from the portfolio_memory dictionary
    #     for asset, values in portfolio_memory.items():
    #         df_portfolio_memory[asset] = values[:-1]

    #     return df_portfolio_memory


# -----------------v1-----------------

# class Portfolio_BBG_window_v1(gym.Env):
#     metadata = {'render.modes': ['human']}

#     def __init__(self,
#                  df,
#                  action_space,
#                  stock_dim,
#                  stock_list,
#                  short_selling_allowed,
#                  hmax: int,
#                  num_stock_shares: list[int],
#                  take_leverage_allowed,
#                  trade_cost_pct,
#                  initial_amount: int,
#                  state_space,
#                  indicators,
#                  reward_scaling,
#                  previous_state=[],
#                  day=0,
#                  initial=True,
#                  print_verbosity=1,
#                  lookback_window=10,
#                  ):
#         # super().__init__()
#         # action space = number of stocks
#         self.terminal = False
#         self.lookback_window = lookback_window
#         self.action_space = action_space
#         self.previous_state = previous_state
#         self.stock_dim = stock_dim
#         self.reward_scaling = reward_scaling

#         self.stock_list = stock_list
#         self.num_stock_shares = num_stock_shares
#         self.short_selling_allowed = short_selling_allowed
#         self.take_leverage_allowed = take_leverage_allowed
#         self.day = day + self.lookback_window

#         self.df = df
#         self.hmax = hmax
#         self.trade_cost_pct = trade_cost_pct
#         self.state_space = state_space
#         self.indicators = indicators
#         self.initial = initial
#         self.initial_amount = initial_amount
#         self.current_amount = self.initial_amount
#         self.data = self.df.loc[self.day -
#                                 self.lookback_window + 1: self.day, :]

#         self.print_verbosity = print_verbosity
#         self.action_space = spaces.Box(
#             low=-1, high=1, shape=(self.action_space,))
#         self.expected_length = 1 + 2 * self.lookback_window + \
#             self.lookback_window * len(self.indicators)
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(self.expected_length,))

#         self.state = self._initiate_state()

#         # Initiate Reward
#         self.cost = 0
#         self.trades = 0
#         self.reward = 0
#         self.episode = 0
#         self.current_step_cost = 0
#         # memorize total balance
#         self.asset_memory = [self.initial_amount+np.sum(
#             np.array(self.num_stock_shares)
#             * np.array(self.state[1:1+self.stock_dim])
#         )]
#         self.rewards_memory = []

#         self.portfolio_return_memory = [0]
#         self.stock_holdings_memory = [self.num_stock_shares]
#         self.actions_memory = []
#         self.state_memory = ([])
#         self.date_memory = [self._get_date()]

#         self.portfolio_memory = {"cash": [self.initial_amount]}
#         for stock, shares in zip(self.stock_list, self.num_stock_shares):
#             self.portfolio_memory[stock] = [shares]

#     def calculate_sharpe_ratio(self):
#         returns = np.array(self.portfolio_return_memory)
#         std_returns = np.std(returns)
#         if std_returns == 0:
#             std_returns = 1e-6
#         return np.mean(returns) / std_returns * np.sqrt(252)

#     def _sell_stock(self, index, action):
#         if self.short_selling_allowed:
#             print("short slel triggered")
#             sell_num_shares = abs(action)
#             sell_amount = (self.state[index + 1] *
#                            sell_num_shares * (1 - self.trade_cost_pct))
#             self.state[0] += sell_amount
#             self.state[index + self.stock_dim + 1] -= sell_num_shares
#             self.cost += (
#                 self.state[index + 1] * sell_num_shares * self.trade_cost_pct
#             )
#             self.trades += 1
#             self.current_amount += sell_amount
#         else:
#             print("ordinary slel triggered")
#             current_shares_owned = self.state[index + self.stock_dim + 1]
#             print("current shares owned: ", current_shares_owned)
#             sell_num_shares = min(current_shares_owned, abs(action))
#             sell_amount = self.state[index + 1] * \
#                 sell_num_shares * (1 - self.trade_cost_pct)
#             self.state[0] += sell_amount
#             self.state[index + self.stock_dim + 1] -= sell_num_shares
#             self.cost += (
#                 self.state[index + 1] * sell_num_shares * self.trade_cost_pct
#             )
#             self.trades += 1
#             self.current_amount += sell_amount
#         self.current_step_cost += self.state[index +
#                                              1] * sell_num_shares * self.trade_cost_pct

#         return sell_num_shares

#     def _buy_stock(self, index, action):
#         if self.take_leverage_allowed:
#             buy_num_shares = action
#             buy_amount = self.state[index + 1] * \
#                 buy_num_shares * (1 + self.trade_cost_pct)
#             self.state[0] -= buy_amount
#             self.state[index + self.stock_dim + 1] += buy_num_shares
#             self.cost += (self.state[index + 1] *
#                           buy_num_shares * self.trade_cost_pct)
#             self.trades += 1
#             self.current_amount -= buy_amount
#         else:
#             # hier fehlt bedingung für fractional shares, aktuell gehen nur ganze
#             available_amount = self.state[0] // (
#                 self.state[index + 1]*(1 + self.trade_cost_pct))
#             buy_num_shares = min(available_amount, action)
#             buy_amount = self.state[index + 1] * \
#                 buy_num_shares * (1 + self.trade_cost_pct)
#             self.state[0] -= buy_amount
#             self.state[index + self.stock_dim + 1] += buy_num_shares
#             self.cost += (self.state[index + 1] *
#                           buy_num_shares * self.trade_cost_pct)
#             self.trades += 1
#             self.current_amount -= buy_amount
#         self.current_step_cost += self.state[index +
#                                              1] * buy_num_shares * self.trade_cost_pct

#         return buy_num_shares

#     def calculate_reward(self, begin_total_asset, end_total_asset):
#         total_asset_reward = end_total_asset - \
#             begin_total_asset - self.current_step_cost
#         return total_asset_reward + self.reward_scaling

#     def step(self, actions):
#         self.terminal = self.day >= len(self.df.index.unique()) - 1
#         if self.terminal:
#             print(f"Terminating on day {self.day} due to condition.")
#             self._handle_terminal_condition()
#         else:
#             most_recent_day_start = -(self.stock_dim + len(self.indicators))
#             most_recent_day_prices = self.state[most_recent_day_start:
#                                                 most_recent_day_start+self.stock_dim]
#             most_recent_day_quantities = self.state[most_recent_day_start +
#                                                     self.stock_dim:most_recent_day_start+2*self.stock_dim]

#             self.current_step_cost = 0
#             actions = actions * self.hmax
#             actions = actions.astype(int)
#             begin_total_asset = self.state[0] + sum(
#                 np.array(most_recent_day_prices) * np.array(most_recent_day_quantities))
#             argsort_actions = np.argsort(actions)
#             sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
#             buy_index = argsort_actions[::-
#                                         1][: np.where(actions > 0)[0].shape[0]]
#             for index in sell_index:
#                 actions[index] = self._sell_stock(index, actions[index]) * (-1)
#             for index in buy_index:
#                 actions[index] = self._buy_stock(index, actions[index])
#             self.actions_memory.append(actions)
#             self.day += 1
#             self.data = self.df.loc[self.day -
#                                     self.lookback_window + 1: self.day, :]
#             self.state = self._update_state()
#             end_total_asset = self.state[0] + sum(
#                 np.array(most_recent_day_prices) * np.array(most_recent_day_quantities))

#             self.asset_memory.append(end_total_asset)
#             self.date_memory.append(self._get_date())
#             portfolio_return = (end_total_asset / begin_total_asset) - 1
#             self.portfolio_return_memory.append(portfolio_return)
#             self.reward = self.calculate_reward(
#                 begin_total_asset, end_total_asset)
#             self.rewards_memory.append(self.reward)
#             self.state_memory.append(self.state)
#             self.stock_holdings_memory.append(
#                 self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)].copy())

#             self.portfolio_memory["cash"].append(self.state[0])
#             for i, stock_name in enumerate(self.stock_list):
#                 self.portfolio_memory[stock_name].append(
#                     self.state[i + self.stock_dim + 1])
#         return self.state, self.reward, self.terminal, False, {}

#     def reset(self, *, seed=None, options=None):
#         self.day = self.lookback_window
#         self.data = self.df.loc[self.day -
#                                 self.lookback_window + 1:self.day, :]

#         self.current_amount = self.initial_amount
#         self.state = self._initiate_state()
#         # print("reset - Initial state:", self.state)
#         if self.initial:
#             self.asset_memory = [
#                 self.initial_amount
#                 + np.sum(
#                     np.array(self.num_stock_shares) *
#                     np.array(self.state[1:self.stock_dim + 1])
#                 )
#             ]
#         else:
#             previous_total_asset = self.previous_state[0] + sum(
#                 np.array(self.state[1:(self.stock_dim + 1)])*np.array(
#                     self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
#             )
#             self.asset_memory = [previous_total_asset]
#         self.cost = 0
#         self.trades = 0
#         self.terminal = False
#         self.rewards_memory = []
#         self.portfolio_return_memory = [0]
#         self.stock_holdings_memory = [self.num_stock_shares]
#         self.actions_memory = []
#         self.date_memory = [self._get_date()]
#         self.episode += 1
#         #! hier von chat.. weiss nicht ob das richtiger reset ist
#         self.portfolio_memory = {"cash": [self.initial_amount]}
#         for stock in self.stock_list:
#             self.portfolio_memory[stock] = [
#                 self.num_stock_shares[self.stock_list.index(stock)]]

#         # chatgpt suggested to convert self. satet to np array
#         return self.state, {}

#     def _initiate_state(self):
#         # Adjusting for lookback period
#         # data_lookback = self.data[self.day - self.lookback_window +1: self.day, :]
#         close_prices = self.data.close.values.tolist()
#         indicators_data = []
#         for indicator in self.indicators:
#             indicators_data.extend(self.data[indicator].values.tolist())
#         if self.initial:
#             # For Initial State for one stock
#             if len(self.df.tic.unique()) > 1:
#                 raise ValueError("Stocks are more than one")
#             else:
#                 state = (
#                     [self.initial_amount] +  # vector of amount of dollar we have
#                     close_prices +
#                     [0]*self.stock_dim*self.lookback_window +
#                     indicators_data
#                 )
#         else:
#             if len(self.df.tic.unique()) > 1:
#                 raise ValueError("Stocks are more than one")
#             else:

#                 if not self.previous_state or len(self.previous_state) < self.expected_length:
#                     print(f"Previous state has not the expected length")
#                     # Handle the initial state or use some default values
#                     state = (
#                         [self.initial_amount] +
#                         close_prices +
#                         [0] * self.stock_dim * self.lookback_window +
#                         indicators_data
#                     )
#                 else:
#                     state = (
#                         [self.previous_state[0]] +
#                         close_prices +
#                         self.previous_state[
#                             (self.stock_dim*self.lookback_window+1): (self.stock_dim * 2 * self.lookback_window + 1)] +
#                         indicators_data
#                     )

#         return state

#     def _update_state(self):
#         close_prices = self.data.close.values.tolist()
#         indicators_data = []
#         for indicator in self.indicators:
#             indicators_data.extend(self.data[indicator].values.tolist())
#         if len(self.df.tic.unique()) > 1:
#             raise ValueError("Stocks are more than one")

#         else:
#             prev_quantities = self.state[-(self.stock_dim * self.lookback_window +
#                                            self.stock_dim):- (self.stock_dim * self.lookback_window)]

#             state = (
#                 [self.state[0]] +  # Retain the initial amount
#                 close_prices +  # Close prices for the lookback window
#                 prev_quantities +
#                 indicators_data  # Indicator values for the lookback window
#             )
#         print("Current state:", state)
#         return np.array(state, dtype=np.float32)

#     def _get_date(self):
#         if len(self.df.tic.unique()) > 1:
#             date = self.data.date.unique()[-1]
#         else:
#             date = self.data.date.iloc[-1]
#         return date

#     def _handle_terminal_condition(self):

#         end_total_asset = self.state[0] + sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
#             self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))
#         df_total_value = pd.DataFrame(self.asset_memory)
#         total_reward = (
#             self.state[0]
#             + sum(
#                 np.array(self.state[1:(self.stock_dim + 1)])
#                 * np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
#             ) - self.asset_memory[0]
#         )
#         df_total_value.columns = ['account_value']
#         df_total_value["date"] = self.date_memory
#         df_total_value['daily_return'] = df_total_value['account_value'].pct_change(
#             1)
#         sharpe = 0
#         if df_total_value['daily_return'].std() != 0:
#             sharpe = (
#                 (252**0.5)*df_total_value['daily_return'].mean() /
#                 df_total_value['daily_return'].std()
#             )
#         df_rewards = pd.DataFrame(self.rewards_memory)
#         df_rewards.columns = ['account_rewards']
#         df_rewards['date'] = self.date_memory[:-1]
#         df_daily_return = pd.DataFrame(self.portfolio_return_memory)
#         df_daily_return.columns = ['daily_return']
#         annualized_return = (
#             ((end_total_asset / self.asset_memory[0]) ** (365 / (self.day - self.lookback_window)))-1)*100
#         if self.episode % self.print_verbosity == 0:
#             self.print_run_outcome(end_total_asset,
#                                    total_reward, df_total_value, sharpe, annualized_return)
#         return self.state, self.reward, self.terminal, False, {}

#     def print_run_outcome(self, end_total_asset, total_reward, df_total_value, sharpe, annualized_return):
#         print(f"day: {self.day}, episode: {self.episode}")
#         print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
#         print(f"end_total_asset: {end_total_asset:0.2f}")
#         print(f"total_reward: {total_reward:0.2f}")
#         print(f"total_cost: {self.cost:0.2f}")
#         print(f"total_trades: {self.trades}")
#         if df_total_value["daily_return"].std() != 0:
#             print(f"Sharpe: {sharpe:0.3f}")
#         print(f"Annual Return: {annualized_return:0.2f}%")
#         print("=================================")

#     def save_state_memory(self):
#         if len(self.df.tic.unique()) > 1:
#             # date and close price length must match actions length
#             date_list = self.date_memory[:-1]
#             df_date = pd.DataFrame(date_list)
#             df_date.columns = ["date"]

#             state_list = self.state_memory
#             df_states = pd.DataFrame(
#                 state_list,
#                 columns=[
#                     "cash",
#                     "Bitcoin_price",
#                     "Gold_price",
#                     "Bitcoin_num",
#                     "Gold_num",
#                     "Bitcoin_Disable",
#                     "Gold_Disable",
#                 ],
#             )
#             df_states.index = df_date.date
#             # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
#         else:
#             date_list = self.date_memory[:-1]
#             state_list = self.state_memory
#             df_states = pd.DataFrame({"date": date_list, "states": state_list})
#         # print(df_states)
#         return df_states

#     def save_asset_memory(self):

#         date_list = self.date_memory[:-1]
#         asset_list = self.asset_memory[:-1]
#         # print(len(date_list))
#         # print(len(asset_list))
#         print(len(date_list))
#         print(len(asset_list))
#         df_account_value = pd.DataFrame(
#             {"date": date_list, "account_value": asset_list}
#         )
#         return df_account_value

#     def save_stock_holdings_memory(self):
#         date_list = self.date_memory[:-1]
#         stock_holdings_list = self.stock_holdings_memory[:-1]
#         print(len(date_list))
#         print(len(stock_holdings_list))
#         df_stock_holdinds = pd.DataFrame(
#             {"date": date_list, "stock_holdings": stock_holdings_list})
#         return df_stock_holdinds

#     def save_return_memory(self):
#         date_list = self.date_memory[:-1]
#         return_list = self.portfolio_return_memory[:-1]
#         print(len(date_list))
#         print(len(return_list))
#         df_return_memory = pd.DataFrame(
#             {"date": date_list, "daily_return": return_list})
#         return df_return_memory

#     def save_action_memory(self):
#         if len(self.df.tic.unique()) > 1:
#             raise ValueError("Stocks are more than one")
#         else:
#             date_list = self.date_memory[:-1]
#             action_list = self.actions_memory
#             print(len(date_list))
#             print(len(action_list))
#             df_actions = pd.DataFrame(
#                 {"date": date_list, "actions": action_list})
#         return df_actions

#     def save_portfolio_memory(self):
#         portfolio_memory = self.portfolio_memory
#         date_list = self.date_memory[:-1]

#         # Create a DataFrame with dates
#         df_portfolio_memory = pd.DataFrame(date_list, columns=["date"])

#         # Add columns from the portfolio_memory dictionary
#         for asset, values in portfolio_memory.items():
#             df_portfolio_memory[asset] = values[:-1]

#         return df_portfolio_memory

#     def _seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def get_env(self):
#         e = DummyVecEnv([lambda: self])
#         obs = e.reset()
#         return e, obs
