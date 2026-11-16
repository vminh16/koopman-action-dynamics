"""Single-factor diagnostic: remove only the executed Gramian penalty."""
import argparse
import experiment as e

original_losses = e.losses

def without_gramian(*args, **kwargs):
    total, prediction, regularizer = original_losses(*args, **kwargs)
    return total - regularizer, prediction, regularizer * 0

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--system',choices=('pendulum','cartpole'),required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=100)
    args=p.parse_args();args.arm='nogram';e.losses=without_gramian;e.train(args)
