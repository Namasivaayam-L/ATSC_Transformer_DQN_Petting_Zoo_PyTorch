import sys, os, logging
from tqdm import tqdm

# Add project root and drl code to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'experiments', 'drl'))

from sumo_rl import parallel_env
import memory
import dqn_double
from utils.update_csv import update_csv
from utils.setup_env import setup_env

# Setup config and logger
config = setup_env()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [drl_double] %(message)s')
logger = logging.getLogger('drl_double')
config['logging'] = logger


def train(config):
    env = parallel_env(
        net_file=config['net_file'],
        route_file=config['route_file'],
        out_csv_name=config['csv_path'],
        use_gui=config['use_gui'],
        num_seconds=config['num_seconds'],
        yellow_time=config['yellow_time'],
        min_green=config['min_green'],
        max_green=config['max_green'],
        reward_fn=config['reward_fn'],
    )
    logger.info(f"Env agents: {env.possible_agents}")

    agent = dqn_double.DoubleDQN(
        num_actions=env.action_spaces[env.possible_agents[0]].n,
        num_states=config['num_states'],
        width=config['width'],
        num_heads=config['num_heads'],
        num_enc_layers=config['num_enc_layers'],
        embedding_dim=config['embedding_dim'],
        num_bins=config['num_bins'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        n_step=config['n_step'],
        target_update_freq=config.get('target_update_freq', 1000),
    )
    logger.info("Initialized DoubleDQN")

    for ep in range(config['num_episodes']):
        logger.info(f"Episode {ep} start")
        state, _ = env.reset()
        terminations = {a: False for a in env.possible_agents}
        step = 0
        while not all(terminations.values()):
            actions = {a: agent.act(state[a]) for a in env.possible_agents}
            next_state, rewards, terminations, truncations, infos = env.step(actions)
            for a in env.possible_agents:
                agent.memory.add(state[a], actions[a], rewards[a], next_state[a])
            state = next_state
            step += 1
        agent.learn()
        logger.info(f"Episode {ep} finished, steps={step}")
        env.save_csv(config['csv_path'], ep)

    env.close()
    logger.info("Training complete")


if __name__ == '__main__':
    train(config)
