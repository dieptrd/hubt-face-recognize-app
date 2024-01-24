import os
import configparser
from pathlib import Path
class AppSettings(configparser.ConfigParser):
    def __init__(self, env="development"):
        super().__init__()
        self.env = env
        self.setting_file = self.get_full_path("./config/settings/", self.env+".ini")
        self.read(self.setting_file)
        self.set_deepface_home()
    def update(self):
        with open(self.setting_file, 'w') as configfile:
            self.write(configfile)
        return 1
    
    def set_deepface_home(self, path=None):
        if not path:
            folder = self.get("GLOBAL", "DEEPFACE_HOME", fallback="./models")
        else:
            folder = path
        folder = self.get_full_path(folder)
        print("Weight folder: ", folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.environ['DEEPFACE_HOME'] = folder
        
    def get_deepface_home(self):
        return os.environ['DEEPFACE_HOME']
    
    def get_full_path(self, dir, *paths):
        path = os.path.join(dir, *paths)
        if(path.startswith('./')):
            parent_dir = os.getcwd()
            subfolder = os.path.join(parent_dir, '_internal')
            if os.path.isdir(subfolder):
                parent_dir = subfolder
            path = os.path.join(parent_dir, path[2:])
        elif(path.startswith('../')):
            parent_dir = os.path.dirname(os.getcwd())
            subfolder = os.path.join(parent_dir, '_internal')
            if os.path.isdir(subfolder):
                parent_dir = subfolder
            path = os.path.join(parent_dir, path[3:])
        return path
    
    def class_name(self, class_name = None):   
        if isinstance(class_name, str):
            class_name = [name.strip() for name in class_name.split(",")]
        
        if isinstance(class_name, list):
            self.set("APPLICATION", "CLASS_NAME", ",".join(class_name))
        
        text = self.get("APPLICATION", "CLASS_NAME", fallback="")
        return text.split(",")

settings = AppSettings()
print("Environment: ", settings.env)