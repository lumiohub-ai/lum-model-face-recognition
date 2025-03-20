import yaml

class Config:
    def __init__(self, config_file='src/face_recognition/cfg/config.yaml'):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            
        # Set all items as attributes of the instance
        for key, value in config.items():
            setattr(self, key, value)
        
    def show_configs(self) -> None:
        # Show configurations to user with colors printings
        for key, value in self.__dict__.items():
            print(f"\033[1m{key}\033[0m: {value}")

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        # Either raise AttributeError or return a default value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")