import random
import math

def draw(weights):
    """
    [float] -> int
    roulette:
    pick an index from the given list of floats proportionally
    to the size of the entry (i.e. normalize to a probability
    distribution and draw according to the probabilities).
    """
    choice = random.uniform(0, sum(weights))
    choice_index = 0
    
    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choice_index
            
        choice_index += 1
        
def normalize(weights):
    """
    normalize to a distribution
    """
    norm = sum(weights)
    return tuple(m / norm for m in weights)
    
def sign(x):
    """
    indicator function
    """
    return 1 if x >= 0 else -1