import jsonargparse
from jsonargparse import ActionConfigFile

parser = jsonargparse.ArgumentParser()

parser.add_argument('--config_file', action=ActionConfigFile, help='Path to the config file')

# config = parser.parse_path('config_file', action=ActionConfigFile, help='Path to the config file')
config = parser.parse_args()

print(config)
