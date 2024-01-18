import os
import configparser
from pathlib import Path
class AppSettings(configparser.ConfigParser):
    def __init__(self, env="development"):
        super().__init__()
        self.env = env
        self.read("./config/settings/"+self.env+".ini")
        self.set_deepface_home()
    def update(self):
        # self.write("./config/settings/"+self.env+".ini")
        return 1
    def set_deepface_home(self, path=None):
        if not path:
            folder = self.get("GLOBAL", "DEEPFACE_HOME", fallback=str(Path.home()))
        else:
            folder = path
        print("Weight folder: ", folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.environ['DEEPFACE_HOME'] = folder
    def get_deepface_home(self):
        return os.environ['DEEPFACE_HOME']
settings = AppSettings()
print("Environment: ", settings.env)