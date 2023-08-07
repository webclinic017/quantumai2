from optuna.visualization import plot_optimization_history, plot_param_importances
import time
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import joblib
import optuna
import numpy as np
import logging
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.integration import SkoptSampler


# Get the Optuna logger
logger = logging.getLogger(optuna.__name__)
logger.setLevel(logging.INFO)

# Create a file handler
handler = logging.FileHandler('optuna_log_Stock_Trading.log')
handler.setLevel(logging.INFO)

# Add the handler to the logger
logger.addHandler(handler)


def optimize_optuna_StockTrading(env_train, env_validate, callbacks, n_trial_runs=10, n_jobs=1):
    best_model = None
    best_value = -np.inf

    def objective(trial):
        print(f"Trial {trial.number} started.")

        start_time = time.time()  # Start the timer
        learning_rate = trial.suggest_float(
            "learning_rate", 0.00001, 0.0001, log=True)
        n_steps = trial.suggest_int("n_steps", 1000, 100000)
        timesteps = trial.suggest_int("total_timesteps", 100000, 3000000)
        gamma = trial.suggest_float("gamma", 0.95, 0.9999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 0.3, log=True)
        max_grad_norm = trial.suggest_float(
            "max_grad_norm", 0.1, 0.4, log=True)
        #!envrionment
        # reward scaling
        reward_scaling = trial.suggest_float(
            "reward_scaling", 1e-5, 1e-3, log=True)
        env_train.reward_scaling = reward_scaling
        # hmax optimization
        hmax = trial.suggest_int("hmax", 20000, 100000)
        env_train.hmax = hmax
        # high_profit_bonus
        high_profit_bonus = trial.suggest_int("high_profit_bonus", 200, 500)
        env_train.high_profit_bonus = high_profit_bonus
        # partial_sell_bonus
        partial_sell_bonus = trial.suggest_int("partial_sell_bonus", 200, 500)
        env_train.partial_sell_bonus = partial_sell_bonus
        # profit_threshold
        profit_threshold = trial.suggest_float(
            "profit_threshold", 0.05, 0.2, log=True)
        env_train.profit_threshold = profit_threshold
        # sharpe_weight scaling
        sharpe_ratio_weight = trial.suggest_float(
            "sharpe_ratio_weight", 0.2, 0.6, log=True)
        env_train.sharpe_ratio_weight = sharpe_ratio_weight
        # loss pentalty weight
        loss_penalty_weight = trial.suggest_float(
            "loss_penalty_weight", 0.1, 0.6, log=True)
        env_train.loss_penalty_weight = loss_penalty_weight
        # env_train.lookback = trial.suggest_int("lookback", 100, 700)
        #  short selling for now
        # short_selling_allowed = trial.suggest_categorical('short_selling_allowed', [True, False])
        # env_train.short_selling_allowed = short_selling_allowed
        # Allow leverage
        # take_leverage_allowed = trial.suggest_categorical('take_leverage_allowed', [True, False])
        # env_train.take_leverage_allowed = take_leverage_allowed

        A2C_PARAMS = {
            "n_steps": n_steps,
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
        }
        # Train the model
        try:
            model = A2C('MlpPolicy', env_train, verbose=0,
                        tensorboard_log="./tensorboard/a2c_florian/", **A2C_PARAMS)
            model.learn(total_timesteps=timesteps, callback=callbacks,
                        tb_log_name="a2c_florian_any_name")
        except Exception as e:
            print(f"Exception during trial {trial.number}: {e}")
            return -np.inf

        # Test the model and return the mean reward
        mean_reward, _ = evaluate_policy(
            model, env_validate, n_eval_episodes=10)
        # Save the model if it's better than the current best
        nonlocal best_model, best_value
        if mean_reward > best_value:
            best_value = mean_reward
            best_model = model
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        print(f"Trial {trial.number} took {elapsed_time} seconds.")
        return mean_reward

    def save_checkpoint(study, trial):
        joblib.dump(study, 'checkpoint.pkl')
    # Create a study object and optimize the objective function
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trial_runs,
                       callbacks=[save_checkpoint], n_jobs=n_jobs)
        # Save the best model to a file
        if best_model is not None:
            best_model.save("best_model_stock_trading")
        else:
            print("No best model to save. All trials may have failed.")
    except Exception as e:
        print(f"Exception during trial : {e}")

    print('Best trial:')
    trial_ = study.best_trial

    print(f'Value: {trial_.value}')

    print('Best hyperparameters:')
    for key, value in trial_.params.items():
        print(f'    {key}: {value}')

    return study


def optimize_optuna(env_train, env_validate, callbacks, n_trial_runs=10):
    best_model = None
    best_value = -np.inf

    def objective(trial):
        print(f"Trial {trial.number} started.")

        start_time = time.time()  # Start the timer
        learning_rate = trial.suggest_float(
            "learning_rate", 0.00001, 0.0001, log=True)
        n_steps = trial.suggest_int("n_steps", 1000, 10000)
        timesteps = trial.suggest_int("total_timesteps", 10000, 100000)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 0.3, log=True)
        max_grad_norm = trial.suggest_float(
            "max_grad_norm", 0.1, 0.4, log=True)
        # envrionment
        reward_scaling = trial.suggest_loguniform("reward_scaling", 1e-5, 1e-3)
        env_train.reward_scaling = reward_scaling
        # env_train.lookback = trial.suggest_int("lookback", 100, 700)
        short_selling_allowed = trial.suggest_categorical(
            'short_selling_allowed', [True, False])
        take_leverage_allowed = trial.suggest_categorical(
            'take_leverage_allowed', [True, False])
        env_train.short_selling_allowed = short_selling_allowed
        env_train.take_leverage_allowed = take_leverage_allowed

        A2C_PARAMS = {
            "n_steps": n_steps,
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
        }
        # Train the model
        # Reset Model
        env_train.reset()
        model = A2C('MlpPolicy', env_train, verbose=0,
                    tensorboard_log="./tensorboard/a2c_florian/", **A2C_PARAMS)
        model.learn(total_timesteps=timesteps, callback=callbacks,
                    tb_log_name="a2c_florian_any_name")

        # Test the model and return the mean reward
        mean_reward, _ = evaluate_policy(
            model, env_validate, n_eval_episodes=10)
        # Save the model if it's better than the current best
        nonlocal best_model, best_value
        if mean_reward > best_value:
            best_value = mean_reward
            best_model = model
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        print(f"Trial {trial.number} took {elapsed_time} seconds.")
        return mean_reward

    def save_checkpoint(study, trial):
        joblib.dump(study, 'checkpoint.pkl')
    # Create a study object and optimize the objective function
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trial_runs,
                       callbacks=[save_checkpoint])
        # Save the best model to a file
        best_model.save("best_model")
    except Exception as e:
        print(f"Exception during trial: {e}")

    print('Best trial:')
    trial_ = study.best_trial

    print(f'Value: {trial_.value}')

    print('Best hyperparameters:')
    for key, value in trial_.params.items():
        print(f'    {key}: {value}')

    return study


def optimize_optuna_FlorianPortfolioEnvMultiReward(env_train, env_validate, callbacks, n_trial_runs=100):
    best_model = None
    best_value = -np.inf

    def objective(trial):
        print(f"Trial {trial.number} started.")

        start_time = time.time()  # Start the timer
        learning_rate = trial.suggest_float(
            "learning_rate", 0.00001, 0.0001, log=True)
        n_steps = trial.suggest_int("n_steps", 1000, 10000)
        timesteps = trial.suggest_int("total_timesteps", 800000, 3000000)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 0.3, log=True)
        max_grad_norm = trial.suggest_float(
            "max_grad_norm", 0.1, 0.4, log=True)
        # envrionment
        reward_scaling = trial.suggest_float(
            "reward_scaling", 1e-5, 1e-3, log=True)
        env_train.reward_scaling = reward_scaling
        # hmax optimization
        hmax = trial.suggest_int("hmax", 40000, 100000)
        env_train.hmax = hmax
        # sharpe_weight scaling
        sharpe_ratio_weight = trial.suggest_float(
            "sharpe_ratio_weight", 0.2, 0.9, log=True)
        env_train.sharpe_ratio_weight = sharpe_ratio_weight
        # loss pentalty weight
        loss_penalty_weight = trial.suggest_float(
            "loss_penalty_weight", 0.1, 0.6, log=True)
        env_train.loss_penalty_weight = loss_penalty_weight
        # env_train.lookback = trial.suggest_int("lookback", 100, 700)
        #! Allow leverage / short selling for now
        # short_selling_allowed = trial.suggest_categorical('short_selling_allowed', [True, False])
        # take_leverage_allowed = trial.suggest_categorical('take_leverage_allowed', [True, False])
        # env_train.short_selling_allowed = short_selling_allowed
        # env_train.take_leverage_allowed = take_leverage_allowed

        A2C_PARAMS = {
            "n_steps": n_steps,
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
        }
        # Train the model
        try:
            model = A2C('MlpPolicy', env_train, verbose=0,
                        tensorboard_log="./tensorboard/a2c_florian/", **A2C_PARAMS)
            model.learn(total_timesteps=timesteps, callback=callbacks,
                        tb_log_name="a2c_florian_any_name")
        except Exception as e:
            print(f"Exception during trial {trial.number}: {e}")
            return None  # or some large negative number

        # Test the model and return the mean reward
        mean_reward, _ = evaluate_policy(
            model, env_validate, n_eval_episodes=10)
        # Save the model if it's better than the current best
        nonlocal best_model, best_value
        if mean_reward > best_value:
            best_value = mean_reward
            best_model = model
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        print(f"Trial {trial.number} took {elapsed_time} seconds.")
        return mean_reward

    def save_checkpoint(study, trial):
        joblib.dump(study, 'checkpoint.pkl')
    # Create a study object and optimize the objective function
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trial_runs,
                       callbacks=[save_checkpoint])
        # Save the best model to a file
        best_model.save("best_model")
    except Exception as e:
        print(f"Exception during trial : {e}")

    print('Best trial:')
    trial_ = study.best_trial

    print(f'Value: {trial_.value}')

    print('Best hyperparameters:')
    for key, value in trial_.params.items():
        print(f'    {key}: {value}')

    return study


def optimize_optuna_BBG_Env(env_train, env_validate, callbacks, n_trial_runs=100):
    best_model = None
    best_value = -np.inf

    def objective(trial):
        print(f"Trial {trial.number} started.")

        start_time = time.time()  # Start the timer
        learning_rate = trial.suggest_float(
            "learning_rate", 0.00001, 0.0001, log=True)
        n_steps = trial.suggest_int("n_steps", 1000, 10000)
        timesteps = trial.suggest_int("total_timesteps", 100000, 120000)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 0.3, log=True)
        max_grad_norm = trial.suggest_float(
            "max_grad_norm", 0.1, 0.4, log=True)
        # envrionment
        reward_scaling = trial.suggest_float(
            "reward_scaling", 1e-5, 1e-3, log=True)
        env_train.reward_scaling = reward_scaling
        # hmax optimization
        hmax = trial.suggest_int("hmax", 40000, 100000)
        env_train.hmax = hmax
        # sharpe_weight scaling
        sharpe_ratio_weight = trial.suggest_float(
            "sharpe_ratio_weight", 0.2, 0.9, log=True)
        env_train.sharpe_ratio_weight = sharpe_ratio_weight
        # loss pentalty weight
        loss_penalty_weight = trial.suggest_float(
            "loss_penalty_weight", 0.1, 0.6, log=True)
        env_train.loss_penalty_weight = loss_penalty_weight
        # env_train.lookback = trial.suggest_int("lookback", 100, 700)
        #! Allow leverage / short selling for now
        # short_selling_allowed = trial.suggest_categorical('short_selling_allowed', [True, False])
        # take_leverage_allowed = trial.suggest_categorical('take_leverage_allowed', [True, False])
        # env_train.short_selling_allowed = short_selling_allowed
        # env_train.take_leverage_allowed = take_leverage_allowed

        A2C_PARAMS = {
            "n_steps": n_steps,
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
        }
        # Train the model
        try:
            model = A2C('MlpPolicy', env_train, verbose=0,
                        tensorboard_log="./tensorboard/a2c_florian/", **A2C_PARAMS)
            model.learn(total_timesteps=timesteps, callback=callbacks,
                        tb_log_name="a2c_florian_any_name")
        except Exception as e:
            print(f"Exception during trial {trial.number}: {e}")
            return None  # or some large negative number

        # Test the model and return the mean reward
        mean_reward, _ = evaluate_policy(
            model, env_validate, n_eval_episodes=10)
        # Save the model if it's better than the current best
        nonlocal best_model, best_value
        if mean_reward > best_value:
            best_value = mean_reward
            best_model = model
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        print(f"Trial {trial.number} took {elapsed_time} seconds.")
        return mean_reward

    def save_checkpoint(study, trial):
        joblib.dump(study, 'checkpoint_bbg.pkl')
    # Create a study object and optimize the objective function
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trial_runs,
                       callbacks=[save_checkpoint])
        # Save the best model to a file
        best_model.save("best_model_bbg")
    except Exception as e:
        print(f"Exception during trial : {e}")

    print('Best trial:')
    trial_ = study.best_trial

    print(f'Value: {trial_.value}')

    print('Best hyperparameters:')
    for key, value in trial_.params.items():
        print(f'    {key}: {value}')

    return study
