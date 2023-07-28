import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from gymnasium.utils import seeding


class FlorianPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,df,action_space,stock_dim,short_selling_allowed,hmax: int,
                 num_stock_shares: list[int],take_leverage_allowed,trade_cost_pct,
                 initial_amount:int,state_space,indicators,reward_scaling,previous_state=[],
                 day=0,initial=True,print_verbosity=10,make_plots:bool = False, model_name="", mode="", iteration=""):
        # super().__init__()
        #action space = number of stocks
        self.terminal = False
        self.action_space = action_space
        self.previous_state = previous_state
        self.model_name = model_name
        self.reward_scaling = reward_scaling
        self.stock_dim = stock_dim
        self.num_stock_shares = num_stock_shares
        self.short_selling_allowed = short_selling_allowed
        self.take_leverage_allowed = take_leverage_allowed
        self.day = day
        self.mode =""
        self.iteration=""
        self.make_plots = make_plots
        self.df = df
        self.hmax = hmax
        self.trade_cost_pct = trade_cost_pct
        self.state_space = state_space
        self.indicators = indicators
        self.initial = initial
        self.initial_amount = initial_amount
        self.data = self.df.loc[self.day,:]
        self.print_verbosity = print_verbosity
        #! je nach Stock oder portfolio anderes definiert
        # self.action_space = spaces.Box(low=0, high=1,shape=(self.action_space,))
        # self.observation_space = spaces.Box(low=0, high=1,shape=(self.state_space,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.state = self._initiate_state()

        #Initiate Reward
        self.cost = 0
        self.trades = 0
        self.reward = 0
        self.episode = 0
        #memorize total balance
        self.asset_memory = [self.initial_amount+np.sum(
            np.array(self.num_stock_shares)
            *np.array(self.state[1:1+self.stock_dim])
            )]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = ([])
        self.date_memory = [self._get_date()]

    def _sell_stock(self, index, action):
        if self.short_selling_allowed:
            sell_num_shares = abs(action)
            sell_amount = (self.state[index + 1]* sell_num_shares* (1 - self.trade_cost_pct))
            self.state[0] += sell_amount
            self.state[index + self.stock_dim + 1] -= sell_num_shares
            self.cost += (
                self.state[index +1] * sell_num_shares * self.trade_cost_pct
            )
            self.trades += 1
        else:
            sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
            sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.trade_cost_pct)
            self.state[0] += sell_amount
            self.state[index + self.stock_dim + 1] -= sell_num_shares
            self.cost += (
                self.state[index +1] * sell_num_shares * self.trade_cost_pct
            )
            self.trades += 1
        return sell_num_shares
    
    def _buy_stock(self, index, action):
        if self.take_leverage_allowed:
            buy_num_shares = action
            buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.trade_cost_pct)
            self.state[0] -= buy_amount
            self.state[index + self.stock_dim + 1] += buy_num_shares
            self.cost += (self.state[index + 1] * buy_num_shares * self.trade_cost_pct)
            self.trades += 1        
        else:
            #hier fehlt bedingung für fractional shares, aktuell gehen nur ganze
            available_amount = self.state[0] // (self.state[index + 1]*(1 + self.trade_cost_pct))
            buy_num_shares = min(available_amount, action)
            buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.trade_cost_pct)
            self.state[0] -= buy_amount
            self.state[index + self.stock_dim + 1] += buy_num_shares
            self.cost += (self.state[index + 1] * buy_num_shares * self.trade_cost_pct)
            self.trades += 1
        return buy_num_shares
    def _make_plots(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        # Execute one time step within the environment
        # return observation, reward, done, info
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            if self.make_plots:
                self._make_plots()
            #! Now df_account_value contains the account values for the entire episode
            #! You can analyze or visualize df_account_value here, or save it to a file    
            df_account_value = self.save_asset_memory()
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))
            df_total_value = pd.DataFrame(self.asset_memory)
            total_reward = (
                self.state[0]
                +sum(
                np.array(self.state[1:(self.stock_dim + 1)])
                *np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
                ) - self.asset_memory[0]
            )
            df_total_value.columns = ['account_value']
            df_total_value["date"] = self.date_memory
            df_total_value['daily_return']=df_total_value['account_value'].pct_change(1)
            if df_total_value['daily_return'].std() != 0:
                sharpe = (
                    (252**0.5)*df_total_value['daily_return'].mean()/ \
                df_total_value['daily_return'].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ['account_rewards']
            df_rewards['date'] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {total_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")
            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()


            return self.state, self.reward, self.terminal, False, {}
        else:
            actions = actions * self.hmax #actions initially is scaled between 0 to 1
            #that could cause issues maybe as fractional is not handled (glaub ich)
            actions = actions.astype(int) #convert into integer because we can't by fraction of shares
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)])
                *np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))
            argsort_actions = np.argsort(actions)
            #! The amount the agent wants to sell can be interpreted as the conviction
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) *(-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")
            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] =self._buy_stock(index, actions[index])
            self.actions_memory.append(actions)
            self.day += 1
            self.data =self.df.loc[self.day,:]
            self.state = self._update_state()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)]) * np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*self.reward_scaling
            self.state_memory.append(self.state)
        return self.state, self.reward, self.terminal, False, {}

    def reset(self,*,seed=None,options=None):
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.state = self._initiate_state()
        if self.initial:
            self.asset_memory = [
                self.initial_amount
                +np.sum(
                np.array(self.num_stock_shares) *np.array(self.state[1:self.stock_dim + 1])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)])*np.array(self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )
            self.asset_memory = [previous_total_asset]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        #chatgpt suggested to convert self. satet to np array
        return np.array(self.state), {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return self.state
    
    def _initiate_state(self):
        if self.initial:
            #For Initial State for one stock
            if len(self.df.tic.unique()) > 1:
                raise ValueError("Stocks are more than one")
            else:
                state = (
                    [self.initial_amount] +  # vector of amount of dollar we have
                    [self.data.close] +
                    [0]*self.stock_dim + 
                    sum([[self.data[indicator]] for indicator in self.indicators], [])
                )
        else:
            if len(self.df.tic.unique()) > 1:
                raise ValueError("Stocks are more than one")
            else:
                state = (
                    [self.previous_state[0]] + 
                    [self.data.close] +
                    self.previous_state[
                        (self.stock_dim+1) : (self.stock_dim *2 +1)] +
                        sum([[self.data[indicator]] for indicator in self.indicators], [])
                )
        return state
    
    def _update_state(self):
        if len(self.df.tic.unique()) >1:
            raise ValueError("Stocks are more than one")
        else:
            state = (
                [self.state[0]] +
                [self.data.close] +
                list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) +
                sum([[self.data[indicator]] for indicator in self.indicators], [])
            )
        return state

    
    
    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date
    #! heree the df are not returned or not saved
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states
    
    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            raise ValueError("Stocks are more than one")
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def get_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
