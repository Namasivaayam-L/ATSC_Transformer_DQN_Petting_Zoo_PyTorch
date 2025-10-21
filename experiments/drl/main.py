import sys, os, random as rd
from pettingzoo.utils import env
from tqdm import tqdm
import pandas as pd  # for loss logging
sys.path.append('/home/namachu/Documents/personal/ATSC_Transformer_DQN_Petting_Zoo_PyTorch/')
from sumo_rl import parallel_env
import dqn as dqn, memory
from utils.update_csv import update_csv
from utils.setup_env import setup_env
from utils.plot import plot_rewards, plot_results, plot_losses

config = setup_env()
ts_signals = None

def train(config):
    env = parallel_env(
        net_file=config["net_file"],
        route_file=config["route_file"],
        num_seconds=config["num_seconds"],
        out_csv_name=config["csv_path"],
        use_gui=config["use_gui"],
        yellow_time=config["yellow_time"],
        min_green=config["min_green"],
        max_green=config["max_green"],
        reward_fn=config["reward_fn"],
    )
    info = {"step": 0, "state": {}, "rewards": {}, "loss": {}}
    global ts_signals
    ts_signals = env.possible_agents
    
    config['logging'].info(f"Traffic signals: {env.possible_agents}")
    config['logging'].info(f"Action: {env.action_space(env.possible_agents[0]).n}")
    config['logging'].info(f"Observation: {env.observation_space(env.possible_agents[0]).shape[0]}")
    
    agents = {
        ts: (
            dqn.DQN(
                ts,
                env.action_space(ts).n,
                env.observation_space(ts).shape[0],
                config["width"],
                config["num_heads"],
                config["num_enc_layers"],
                config["embedding_dim"],
                config["batch_size"],
                config["gamma"],
                config["learning_rate"],
                config['model_path'],
                config['fine_tune_model_path'],
                config['logging']
            )
        )
        for ts in env.possible_agents
    }
    # create separate replay buffer per agent
    experience_replay = {ts: memory.Memory(config["buffer_size"]) for ts in env.possible_agents}
    for ep in tqdm(range(0, config["num_episodes"]), desc="Running..", unit="epsiode"):
        config['logging'].info(f"Episode: {ep}")
        config['logging'].info(f"Epsilon: {config['epsilon']}")
        state, _ = env.reset()
        # config['logging'].debug(f"State: {state.shape}")
        terminations = {a: False for a in agents}
        epsilon = max(config["epsilon"] * config["decay"]**ep, config["min_epsilon"]) #update epsilon every episode

        while not all(terminations.values()):
            config['logging'].debug(f"State: {state}")
            actions = {ts: agents[ts].act(state[ts], epsilon) for ts in env.possible_agents}
            config['logging'].debug(f"Actions: {actions}")
            next_state, rewards, terminations, truncations, infos = env.step(actions)
            info["state"].update(state)
            info["rewards"].update(rewards)
            config['logging'].debug(f"Rewards: {rewards}")
            if all(terminations.values()):
                # per-agent prioritized replay with loss tracking
                losses = []
                for ts, agent in agents.items():
                    exp = experience_replay[ts].sample(config["batch_size"])
                    indices, new_prios, loss_val = agent.learn(ts, ep, exp)
                    experience_replay[ts].update_priorities(indices, new_prios)
                    info["loss"][ts] = loss_val
                    losses.append(loss_val)
                # compute global loss average
                info["loss"]["global"] = sum(losses) / len(losses) if losses else 0.0
                break
            for ts in env.possible_agents:
                experience_replay[ts].add(state[ts], actions[ts], rewards[ts], next_state[ts])
            config['logging'].debug(f"Added to experience replay")
            state = next_state
            update_csv(info, ep, env.possible_agents, config['output_path'])
            info["step"] += 1
            tqdm.write(f"Progress: {info['step']}/{config['num_seconds']}", end="\r")
        config['logging'].info(f"Episode {ep} finished")
        env.save_csv(config['csv_path'], ep)
        # write loss CSV per episode
        loss_df = pd.DataFrame([info["loss"]])
        loss_df["ep"] = ep
        mode = "w" if ep == 0 else "a"
        loss_df.to_csv(config["output_path"] + "loss.csv", index=False, mode=mode, header=(mode == "w"))
    env.close()
    
train(config)

plot_rewards(config['output_path']+'rewards.csv', ts_signals)
plot_losses(config['output_path']+'loss.csv', ts_signals)
num_eps = int(config["num_episodes"])
ep = rd.randint(int(num_eps*0.75), num_eps-1)
print(f'{config["csv_path"]}{ep}ep.csv')
plot_results([f'{config["output_path"]}csv/{ep}ep.csv'])

config["fine_tune"] = True
config["num_episodes"] =  config["test_num_episodes"]
config["fine_tune_model_path"] = f'{config["output_path"]}models/'
config["output_path"] = f'{config["output_path"]}test/{config["test_num_episodes"]}ep/'
config['csv_path'] = f'{config["output_path"]}csv/'

os.makedirs(config["csv_path"], exist_ok=True)
os.makedirs(config["output_path"], exist_ok=True) 

train(config)

plot_rewards(config['output_path']+'rewards.csv', ts_signals)
plot_losses(config['output_path']+'loss.csv', ts_signals)
test_num_eps = int(config["num_episodes"])
test_ep = rd.randint(int(test_num_eps*0.75), test_num_eps-1)
print(f'{config["csv_path"]}{test_ep}ep.csv')
plot_results([f'{config["output_path"]}csv/{test_ep}ep.csv'])