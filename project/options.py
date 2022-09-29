import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
parser.add_argument("--coef", type=float, default=100.0, help="Coefficient for L1_loss")
'''
parser.add_argument("--lr_Dp", type=float, default=0.03, help="Learning rate (person discriminator)")
parser.add_argument("--lr_Db", type=float, default=0.03, help="Learning rate (background discriminator)")
parser.add_argument("--lr_G", type=float, default=0.1, help="Learning rate (generator)")
'''
opt = parser.parse_args()