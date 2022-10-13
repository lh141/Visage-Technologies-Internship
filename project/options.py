import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
        self.parser.add_argument("--coef", type=float, default=100.0, help="Coefficient for L1_loss")
        self.parser.add_argument("--split", type=float, default=0.8, help="Percentage of training set used for training (the rest is for validation)")
        self.parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs")
        '''
        self.parser.add_argument("--lr_Dp", type=float, default=0.03, help="Learning rate (person discriminator)")
        self.parser.add_argument("--lr_Db", type=float, default=0.03, help="Learning rate (background discriminator)")
        self.parser.add_argument("--lr_G", type=float, default=0.1, help="Learning rate (generator)")
        '''
        self.initialized = True
    
    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        return opt