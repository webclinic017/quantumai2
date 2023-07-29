
class DRLAgent:


    def __init__(self, env):
        self.env = env


    def DRL_prediction(model, environment, deterministic=True):
            test_env, test_obs = environment.get_env()
            """make a prediction"""
            account_memory = []
            actions_memory = []
            #         state_memory=[] #add memory pool to store states
            test_env.reset()
            for i in range(len(environment.df.index.unique())):
                action, _states = model.predict(test_obs, deterministic=deterministic)
                # account_memory = test_env.env_method(method_name="save_asset_memory")
                # actions_memory = test_env.env_method(method_name="save_action_memory")
                test_obs, rewards, dones, info = test_env.step(action)
                #!FLO: second last iteration (-2) weil sich am letzten Punkt nix ändert
                if i == (len(environment.df.index.unique()) - 2):
                    account_memory = test_env.env_method(method_name="save_asset_memory")
                    actions_memory = test_env.env_method(method_name="save_action_memory")
                #                 state_memory=test_env.env_method(method_name="save_state_memory") # add current state to state memory
                if dones[0]:
                    print("hit end!")
                    break
            return account_memory[0], actions_memory[0]
    

    def DRL_prediction_flo(model, environment, deterministic=True):
            test_env, test_obs = environment.get_env()
            """make a prediction"""
            account_memory = []
            actions_memory = []
            return_memory = []
            #         state_memory=[] #add memory pool to store states
            test_env.reset()
            for i in range(len(environment.df.index.unique())):
                action, _states = model.predict(test_obs, deterministic=deterministic)
                # account_memory = test_env.env_method(method_name="save_asset_memory")
                # actions_memory = test_env.env_method(method_name="save_action_memory")
                test_obs, rewards, dones, info = test_env.step(action)
                #!FLO: second last iteration (-2) weil sich am letzten Punkt nix ändert
                if i == (len(environment.df.index.unique()) - 2):
                    account_memory = test_env.env_method(method_name="save_asset_memory")
                    actions_memory = test_env.env_method(method_name="save_action_memory")
                    return_memory = test_env.env_method(method_name="save_return_memory")
                #                 state_memory=test_env.env_method(method_name="save_state_memory") # add current state to state memory
                if dones[0]:
                    print("hit end!")
                    break
            return account_memory[0], actions_memory[0],return_memory[0]