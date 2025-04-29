import sys, os, logging
import torch
from tqdm import tqdm

# Set up paths to import modules
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'experiments', 'drl'))

from utils.setup_env import setup_env
import dqn_functorch

# Configure logging
config = setup_env()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [drl_functorch] %(message)s')
logger = logging.getLogger('drl_functorch')
config['logging'] = logger


def train_functorch(config):
    logger.info("Initializing functional DQN agents...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_agents = config.get('num_agents', 4)
    net_args = dict(
        num_states=config['num_states'],
        num_bins=config['num_bins'],
        num_actions=config.get('num_actions', env.action_spaces[env.possible_agents[0]].n if False else 2),
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_enc_layers=config['num_enc_layers'],
        width=config['width'],
    )
    func, params, buffers = dqn_functorch.initialize_agents(num_agents, device, **net_args)
    logger.info(f"Created {num_agents} functional agents")

    # Dummy batch for smoke test
    batch_size = config['batch_size']
    states = torch.randn(batch_size, net_args['num_states'], device=device)
    actions = torch.randint(0, net_args['num_actions'], (batch_size,), device=device)
    rewards = torch.randn(batch_size, device=device)
    next_states = torch.randn(batch_size, net_args['num_states'], device=device)
    batch = (states, actions, rewards, next_states)

    new_params, new_buffers = dqn_functorch.multiagent_update(
        params, buffers, batch, func,
        lr=config['learning_rate'], gamma=config['gamma']
    )
    logger.info("Completed a multi-agent update stub.")

if __name__ == '__main__':
    train_functorch(config)
