def load_config(config_file):
    import json
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def log_event(event_message):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(event_message)