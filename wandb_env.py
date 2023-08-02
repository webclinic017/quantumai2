import os
import wandb
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from wandb.integration.sb3 import WandbCallback


os.environ["WANDB_API_KEY"] = "7f0155c08333a1cd3fc2bfaae555cd38344b421d"
wandb.login()


def wandb_opt(environment, train_df, project_name, state_space, stock_list, indicators, stock_dimension=1):
    sweep_config = {
        'method': 'random',  # or 'grid' or 'bayes'
        'metric': {
            'name': 'cumulative_reward',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 0.00001,
                'max': 0.0001
            },
            'n_steps': {
                'min': 1000,
                'max': 10000
            },
            'hmax': {
                'min': 30000,
                'max': 80000
            },
            

                "gamma": {
                'min': 0.9,
                'max': 0.9999
                },
                "gae_lambda": {
                    'min': 0.9,
                'max': 1.0
                },
                "ent_coef": {
                    'min': 0.01,
                'max': 0.1
                },
                "vf_coef": {
                            'min': 0.1,
                'max': 0.3
                },
                "max_grad_norm": {
                    'min': 0.1,
                'max': 0.4
                },
                "reward_scaling": {
                        'min': 1e-5,
                'max': 1e-3
                },
                "sharpe_ratio_weight": {
                        'min': 0.1,
                'max': 0.95
                },
                "loss_penalty_weight": {
                        'min': 0.1,
                'max': 0.95
                }
        
            
            # fill the rest of your hyperparameters
        }
    }


    remaining_kwargs = {
        "initial_amount": 1000000, 
        "trade_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension,
        "stock_list": stock_list,
        "indicators": indicators, 
        "action_space": stock_dimension,
        "short_selling_allowed": True,
        "take_leverage_allowed": True,
        # Not for optimization
        "make_plots": False,
        "num_stock_shares": [0], #number of initioal shares
        "model_name": "A-2C",
        "mode": "training", #can be anything, just for plots
        "iteration": "1000"#can be anything, just for plots
        
    }
    def train():
        # Initialize a new wandb run
        run = wandb.init()

        # Get hyperparameters from the config
        config = wandb.config



        e_train_gym_wandb = environment(df=train_df,reward_scaling=config.reward_scaling, hmax=config.hmax, sharpe_ratio_weight=config.sharpe_ratio_weight, loss_penalty_weight=config.loss_penalty_weight, **remaining_kwargs)
        env_train, _ = e_train_gym_wandb.get_env()
        env_train = VecNormalize(env_train)

        # Create and train the model
        model = A2C('MlpPolicy', env_train, verbose=0, tensorboard_log="./tensorboard/a2c_florian/", learning_rate=config.learning_rate, n_steps=config.n_steps, gamma=config.gamma, gae_lambda=config.gae_lambda, ent_coef=config.ent_coef, vf_coef=config.vf_coef, max_grad_norm=config.max_grad_norm)
        try:
            model.learn(total_timesteps=1000000, callback=WandbCallback())
        except Exception as e:
            wandb.log({"run_error": str(e)})
            run.finish()
            return

        # Save the model
        os.makedirs("/wanddb_models/", exist_ok=True)
        model.save(f"/wanddb_models/model_{run.id}")

        # Close the run
        run.finish()


    sweep_id = wandb.sweep(sweep_config, project="bloomberg")
    wandb.agent(sweep_id, train)