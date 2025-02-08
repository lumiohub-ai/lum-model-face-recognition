import yaml

class Config:
    def __init__(self, config_file='src/face_recognition/cfg/config.yaml'):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Set all items as attributes of the instance
        for key, value in config.items():
            setattr(self, key, value)

    def __getattr__(self, item):
        return self.cfg[item]