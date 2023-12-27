import configparser

class AppSettings(configparser.ConfigParser):
    def __init__(self, env="development"):
        super().__init__()
        self.env = env
        self.read("./config/settings/"+self.env+".ini")
    def update(self):
        # self.write("./config/settings/"+self.env+".ini")
        return 1
settings = AppSettings()
print("Environment: ",settings.env)