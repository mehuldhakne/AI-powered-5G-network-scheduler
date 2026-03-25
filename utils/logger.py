import os

class Logger:
    def __init__(self, log_file="logs/train.log"):
        os.makedirs("logs", exist_ok=True)
        self.log_file = log_file

    def log(self, message):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
