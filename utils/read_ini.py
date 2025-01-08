import configparser

def load_config(ini_file):
    """Loads configuration parameters from an INI file.

    Args:
        ini_file (str): Path to the INI configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(ini_file)

    # Directly access and store values in a dictionary.  No intermediate variables.
    cfg = {  # More concise naming
        "net_file": config["Sumo"]["net_file"],
        "route_file": config["Sumo"]["route_file"],  # Added route_file
        "output_path": config["Sumo"]["out_dir"],
        "csv_path": config["Sumo"]["out_dir"] + "csv/",
        "model_path": config["Sumo"]["out_dir"] + "models/",
        "log_path": config["Sumo"]["out_dir"] + "logs/",
        "single_agent": config["Sumo"].getboolean("single_agent"), # Added single_agent
        "use_gui": config["Sumo"].getboolean("use_gui"),       # Added use_gui
        "num_seconds": config["Sumo"].getint("num_seconds"),   # Added num_seconds
        "yellow_time": config["Sumo"].getint("yellow_time"),   # Added yellow_time
        "min_green": config["Sumo"].getint("min_green"),       # Added min_green
        "max_green": config["Sumo"].getint("max_green"),       # Added max_green
        "reward_fn": config["Sumo"]["reward_fn"],             # Added reward_fn
        "fine_tune_model_path": config["Model"]["fine_tune_model_path"] if config["Model"].getboolean("fine_tune") else None,
        "batch_size": config["Model"].getint("batch_size"),
        "gamma": config["Model"].getfloat("gamma"),
        "learning_rate": config["Model"].getfloat("learning_rate"),
        "num_episodes": config["Model"].getint("num_episodes"),
        "num_bins": config["Sumo"].getint("num_bins"),
        "buffer_size": config["Memory"].getint("buffer_size"),
        "num_states": config["Sumo"].getint("num_states"),
        "num_layers": config["Model"].getint("num_layers"),
        "width": config["Model"].getint("width"),
        "num_heads": config["Model"].getint("num_heads"),
        "embedding_dim": config["Model"].getint("embedding_dim"),
        "num_enc_layers": config["Model"].getint("num_enc_layers"),
        "epsilon": config["Model"].getfloat("epsilon"),
        "min_epsilon": config["Model"].getfloat("min_epsilon"),
        "decay": config["Model"].getfloat("decay"),
        "model_name": config["Model"]["name"] #Added model_name
    }

    return cfg # Return the dictionary

