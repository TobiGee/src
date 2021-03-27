import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train', help='train or test?')
    parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')
    parser.add_argument('--dataset', type=str, default='eg', help='dataset_name')
    return check_args(parser.parse_args())



def test():
    None
def train():
    None
if __name__=="__main__":
    print('LEts go!')
    args = parse_args()
