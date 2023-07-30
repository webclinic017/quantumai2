from optuna.visualization import plot_optimization_history, plot_param_importances
import time
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import joblib
import optuna




def optimize_optuna(env_train, env_validate, callbacks, n_trial_runs=10):
    def objective(trial):
        start_time = time.time()  # Start the timer
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_int("n_steps", 100, 5000)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 0.5, log=True)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 0.6, log=True)
        #envrionment
        reward_scaling = trial.suggest_loguniform("reward_scaling", 1e-5, 1e-3)
        env_train.reward_scaling = reward_scaling
        # env_train.lookback = trial.suggest_int("lookback", 100, 700)
        short_selling_allowed = trial.suggest_categorical('short_selling_allowed', [True, False])
        take_leverage_allowed = trial.suggest_categorical('take_leverage_allowed', [True, False])
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
        model = A2C('MlpPolicy', env_train, verbose=0, tensorboard_log="./tensorboard/a2c_florian/", **A2C_PARAMS)
        model.learn(total_timesteps=50000, callback=callbacks, tb_log_name="a2c_florian_any_name")
      


        # Test the model and return the mean reward
        mean_reward, _ = evaluate_policy(model, env_validate, n_eval_episodes=10)
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        print(f"Trial {trial.number} took {elapsed_time} seconds.")
        return mean_reward

    def save_checkpoint(study, trial):
        joblib.dump(study, 'checkpoint.pkl')
    # Create a study object and optimize the objective function
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trial_runs,callbacks=[save_checkpoint])
    except Exception as e:
        print(f"Exception during trial: {e}")
 


    print('Best trial:')
    trial_ = study.best_trial

    print(f'Value: {trial_.value}')

    print('Best hyperparameters:')
    for key, value in trial_.params.items():
        print(f'    {key}: {value}')

