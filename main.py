import argparse
from model import IF4SR
from utils import data_partition

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='grocery', help='dataset name')
parser.add_argument('--maxlen', type=int, default=50, help='max length of sequence')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization')
parser.add_argument('--hidden_units', type=int, default=50, help='hidden dimension')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')

args = parser.parse_args()


if __name__ == '__main__':

    train, valid, test, user_num, item_num = data_partition(args.dataset)

    model = IF4SR(args, item_num).to(args.device)