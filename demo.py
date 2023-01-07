import matplotlib.pyplot as plt
import LS
import time

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

N = range(1, 6)
batches = []
for i in N:
    filename = './data/cifar-10-batches-py/data_batch_{}'.format(i)
    batches.append(unpickle(filename))
tb = unpickle('./data/cifar-10-batches-py/test_batch')
train_batch = [[], []]
for batch in batches:
    train_batch[0] += list(batch[b'data']) # images
    train_batch[1] += list(batch[b'labels']) # class
test_batch = []
test_batch.append(tb[b'data']) # images
test_batch.append(tb[b'labels']) # class
class_names = unpickle('./data/cifar-10-batches-py/batches.meta')[b'label_names']

start_time = time.time()
wt = LS.LSLearn(train_batch, class_names)
print('training time was {}'.format(time.time() - start_time))
correct = LS.LSPredict(test_batch, wt, class_names)
print('correct percentage is {}'.format(correct))