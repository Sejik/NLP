
from NLP.config import args
from NLP.learn.experiment import Experiment
from NLP.learn.mode import Mode

if __name__ == "__main__":
    experiment = Experiment(Mode.TRAIN, args.config(mode=Mode.TRAIN))
    experiment()
