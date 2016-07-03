import sys

class Stump:
    def __init__(self):
        self._gt_label = None
        self._lt_label = None
        self.split_threshold = None
        self.split_feature= None
        
    def classify(self, point):
        if point[self.split_feature] >= self.split_threshold:
            return self._gt_label
        else:
            return self._lt_label
            
    def __call__(self, point):
        return self.classify(point)
        

def majorityVote(data):
    ''' Comput the majority of the class labels in the given data set. '''
    labels = [label for (pt, label) in data]
    try:
        return max(set(labels), key = labels.count)
    except:
        return -1
        
def min_label_error_of_hypothesis_and_negation(data, h):
    '''
    g_error: error of great predict method, which for a sample if it's split_feature
             value great than split_threshold, it's predict label=1
    l_error: error of less predict method, which for a sample if it's split_feature
             value less than split_threshold, it's predict label=1
    '''
    posData, negData = ([(x, y) for (x, y) in data if h(x) == 1],
                        [(x, y) for (x, y) in data if h(x) == -1],)
            
    g_error = sum(y == -1 for (x, y) in posData) + sum( y == 1 for (x, y) in negData)
    l_error = sum(y == 1 for (x, y) in posData) + sum( y == 1 for (x, y) in negData)
    return min(g_error, l_error) / len(data)
    

def bestThreshold(data, index, errorFunction):
   '''Compute best threshold for a given feature. Returns (threshold, error)'''

   thresholds = [point[index] for (point, label) in data]
   def makeThreshold(t):
      return lambda x: 1 if x[index] >= t else -1

   errors = [(threshold, errorFunction(data, makeThreshold(threshold))) for threshold in thresholds]
   return min(errors, key=lambda p: p[1])


def defaultError(data, h):
   return min_label_error_of_hypothesis_and_negation(data, h)


def buildDecisionStump(drawExample, errorFunction=defaultError, debug=True):
   # find the index of the best feature to split on, and the best threshold for
   # that index

   data = [drawExample() for _ in range(500)]

   bestThresholds = [(i,) + bestThreshold(data, i, errorFunction) for i in range(len(data[0][0]))]
   feature, thresh, _ = min(bestThresholds, key = lambda p: p[2])

   stump = Stump()
   stump.split_feature = feature
   stump.split_threshold = thresh
   stump._gt_label = majorityVote([x for x in data if x[0][feature] >= thresh])
   stump._lt_label = majorityVote([x for x in data if x[0][feature] < thresh])

   if debug:
      sys.stderr.write('Feature: %d, threshold: %d, %s\n' % (feature, thresh, '+' if stump._gt_label == 1 else '-'))

   return stump