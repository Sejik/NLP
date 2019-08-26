
from NLP.config import args
from NLP.learn.experiment import Experiment
from NLP.learn.mode import Mode

if __name__ == "__main__":
    experiment = Experiment(Mode.ALL_IN_ONE, args.config(mode=Mode.ALL_IN_ONE))
    experiment()
