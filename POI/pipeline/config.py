import argparse

def parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_pretrained_model", default="roberta", type=str)
    parser.add_argument("--image_pretrained_model", default="InceptionV3", type=str)
    parser.add_argument("--optimizer", default="sgd", type=str)  # sgd or adam
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--loss", default="cc", type=str)  # cc or fl
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--image_size", default=299, type=int)
    parser.add_argument("--max_length", default=50, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--validation_size", default=0.1, type=float)
    parser.add_argument("--seed", default=1011, type=int)

    return parser.parse_args()

if __name__ == '__main__':

    args = parser()
    args = vars(args)
    print(args)