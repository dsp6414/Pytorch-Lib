

import numpy as np

from torch.autograd import Variable



def predict_batch(net, inputs):
    v = Variable(inputs.cuda(), volatile=True)
    return net(v).data.cpu().numpy()


def get_probabilities(model, loader):
    model.eval()
    return np.vstack(predict_batch(model, data[0]) for data in loader)

