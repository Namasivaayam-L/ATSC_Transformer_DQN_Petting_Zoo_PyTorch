import logging, os, sys, shutil
from utils.read_ini import load_config

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s", datefmt="%H:%M:%S")

def setup_env():
    ini_file = "experiments/trf_multi_agent/dqn/config.ini"
    config = load_config(ini_file) # Use the function from read_ini.py
    config['logging'] = logging
    logging.info(f"Creating output directory: {config['output_path']}")
    os.makedirs(config["output_path"], exist_ok=True)
    logging.info(f"Creating CSV directory: {config['csv_path']}")
    os.makedirs(config["csv_path"], exist_ok=True)
    logging.info(f"Creating model directory: {config['model_path']}")
    os.makedirs(config["model_path"], exist_ok=True)
    logging.info(f"Copying config file '{ini_file}' to '{config['output_path'] + 'config.ini'}'")
    shutil.copyfile(ini_file, config["output_path"] + "config.ini")
    logging.info(f"Copying DQN script '{os.getcwd()}/experiments/trf_multi_agent/dqn/dqn.py' to '{config['model_path'] + 'model.py'}'")
    shutil.copyfile(os.getcwd() + "/experiments/trf_multi_agent/dqn/dqn.py", config["model_path"] + "model.py")
    logging.info("Environment setup complete.")

    return config
