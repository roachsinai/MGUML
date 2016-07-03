import random
import boosting
from utils import sign

def error(h, data):
    return sum(1 for x,y in data if h(x) != y) / len(data)
    

def runAdult():
    from data import adult
    from decisionstump import buildDecisionStump
    train, test = adult.load()   
    weakLearner = buildDecisionStump
    rounds = 20
    
    h = boosting.boost(train, weakLearner, rounds)
    print("Training error: %G" % error(h, train))
    print("Test error: %G" % error(h, test))


if __name__ == "__main__":
    runAdult()