

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy
from utils_training import run_taskset
from Utils import get_optim
from Models.model import EncoderClassifier
from global_settings import * # sets the device globally


# knn from https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3
class NN():

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]


def get_features(loader, model: EncoderClassifier):
    ys = []
    xs = []
    with torch.no_grad():
        for x_, y, t in loader:
            x_ = x_.to(device)
            ys.append(y)
            f = model.encoder(x_)
            xs.append(f.to('cpu'))
    xs = torch.cat(xs).squeeze().numpy()
    ys = torch.cat(ys).numpy()
    return xs, ys


def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist

class KNN(NN):

    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner



def probe_knn(taskset_train, taskset_test, model: EncoderClassifier, nb_classes, *args, **kwargs):
    loader = DataLoader(taskset_train, batch_size=512)
    print('testing knn')
    xs, ys = get_features(loader, model)
    # X=torch.tensor(xs).to(device)
    # Y=torch.tensor(ys).to(device)
    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(xs, ys)
    # knn = KNN(k=100)
    # knn.train(X,Y)
    loader = DataLoader(taskset_test, batch_size=512)
    testFeat, testLabels = get_features(loader, model)
    # accs=[]
    # with torch.no_grad():
    #     for x_,y,t in loader:
    #         x_ = x_.to(device)
    #         # ys.append(y)
    #         f = model.encoder(x_)
    #         y_hat = knn.predict(f.squeeze())
    #         acc = np.sum(y==y_hat.cpu().numpy())/len(y)
    #         accs.append(acc)
    # acc = np.mean(accs)
    acc = neigh.score(testFeat, testLabels)
    return acc  # np.mean(accs)


def representation_eval(
    model, full_tr_dataset, full_te_dataset, optim_name, nb_eval_epoch
):
    print("#########################################")
    print("Eval on 10 way classification")

    # change head
    original_head = deepcopy(model.head)

    # freeze model
    for param in model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        model.head = nn.Linear(50, 10).to(device)

    opt_eval = get_optim(model.head, name=optim_name)
    for epoch_val in range(nb_eval_epoch):
        print(f"Eval epoch {epoch_val}")
        run_taskset(full_tr_dataset, model, opt=opt_eval)

        print(f"Test Eval epoch {epoch_val}")
        test_acc = run_taskset(full_te_dataset, model, opt=None)

    # put back head
    with torch.no_grad():
        model.head = original_head

    # unfreeze model
    for param in model.parameters():
        param.requires_grad = True
    print("#########################################")

    return test_acc