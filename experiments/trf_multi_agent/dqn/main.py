import sys
from tqdm import tqdm
sys.path.append('/content/drive/MyDrive/atsc-fyp-colab/')
from sumo_rl import parallel_env
import dqn, memory
from utils.update_csv import update_csv
from utils.setup_env import setup_env
from utils.plot import plot_rewards

config = setup_env()

env = parallel_env(
    net_file=config["net_file"],
    route_file=config["route_file"],
    out_csv_name=config["csv_path"],
    use_gui=config["use_gui"],
    num_seconds=config["num_seconds"],
    yellow_time=config["yellow_time"],
    min_green=config["min_green"],
    max_green=config["max_green"],
    reward_fn=config["reward_fn"],
)

info = {"step": 0, "state": {}, "rewards": {}}
agents = {
    ts: dqn.DQN(
        ts,
        env.action_spaces[ts].n,
        config["num_states"],
        config["width"],
        config["num_heads"],
        config["num_enc_layers"],
        config["embedding_dim"],
        config["num_bins"],
        config["batch_size"],
        config["gamma"],
        config["learning_rate"],
        config['model_path'],
        config['fine_tune_model_path'],
    )
    for ts in env.possible_agents
}

experience_replay = memory.Memory(config["buffer_size"])
for ep in tqdm(range(0, config["num_episodes"]), desc="Running..", unit="epsiode"):
    config['logging'].info(f"Episode: {ep}")
    config['logging'].info(f"Epsilon: {config['epsilon']}")
    state, _ = env.reset()
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
            for key in agents.keys():
                agents[key].learn(experience_replay.sample(config["batch_size"]))
            break
        for ts in env.possible_agents:
            experience_replay.add(state[ts], actions[ts], rewards[ts], next_state[ts])
        config['logging'].debug(f"Added to experience replay")
        state = next_state
        update_csv(info, ep, config['output_path'])
        info["step"] += 1
        tqdm.write(f"Progress: {info['step']}/{config['num_seconds']}", end="\r")
    config['logging'].info(f"Episode {ep} finished")
    env.save_csv(config['csv_path'], ep)
    env.close()

plot_rewards(config['output_path']+'rewards.csv')