import logging
import os

class LoggerManager:
    def __init__(self, antibiotic, fold, save_path, train=True):
        self.logger = logging.getLogger(f'{antibiotic}_fold_{fold}')
        self.logger.setLevel(logging.INFO)
        if train:
            fh = logging.FileHandler(os.path.join(save_path, f'{antibiotic}_fold_{fold}.log'))
        else:
            fh = logging.FileHandler(os.path.join(save_path, f'{antibiotic}_Test_Results_fold_{fold}.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def close(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def log(self, message):
        self.logger.info(message)