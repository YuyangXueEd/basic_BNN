import numpy as np
import torch
import torchbnn as bnn

##################
## Dataset Class##
##################

class IrisDataset(torch.utils.data.Dataset):
    def __init__(self, src_file, num_rows=None):
        # like 5.0, 3.5, 1.3, 0.3, 0
        tmp_x = np.loadtxt(src_file,
                           max_rows=num_rows,
                           usecols=range(0, 4),
                           delimiter=" ",
                           skiprows=0,
                           dtype=np.float32)
        tmp_y = np.loadtxt(src_file,
                           max_rows=num_rows,
                           usecols=4,
                           delimiter=" ",
                           skiprows=0,
                           dtype=np.int64)

        self.x_data = torch.tensor(tmp_x, dtype=torch.float32)
        self.y_data = torch.tensor(tmp_y, dtype=torch.int64)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx]
        spcs = self.y_data[idx]
        sample = {'predictors': preds, 'species': spcs}

        return sample

#################
##Network Class##
#################

class BayesianNet(torch.nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.hid1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=4, out_features=100)
        self.out = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=3)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = self.out(z)
        return z

##############
##Evaluation##
##############

def accuracy(model, dataset):
    dataldr = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    n_correct = 0; n_wrong = 0
    for (_, batch) in enumerate(dataldr):
        X = batch['predictors']
        Y = batch['species']
        with torch.no_grad():
            output = model(X)

        big_idx = torch.argmax(output)
        if big_idx == Y:
            n_correct += 1
        else:
            n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc


def accuracy_quick(model, dataset):
    n = len(dataset)
    X = dataset[0:n]['predictors'].cuda()
    Y = torch.flatten(dataset[0:n]['species']).cuda()

    with torch.no_grad():
        output = model(X)

    arg_maxs = torch.argmax(output, dim=1)
    num_correct = torch.sum(Y==arg_maxs)
    acc = (num_correct * 1.0 / len(dataset))
    return acc.item()

########
##main##
########

def main():
    print("Begin Bayesian NN Iris demo\n")

    ## preparation
    np.random.seed(1)
    torch.manual_seed(1)
    np.set_printoptions(precision=4, suppress=True, sign=" ")
    np.set_printoptions(formatter={'float':'{: 0.4f}'.format})
    print("\n")

    ## load training data
    print("Creating Iris training dataset and dataloader\n")
    train_file = "Data/Iris_train.txt"
    train_ds = IrisDataset(train_file, 120)

    batch_size = 4
    train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    ## init network
    net = BayesianNet().cuda()

    ## training
    max_epochs = 100
    ep_log_interval = 10

    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    print("\nbatch_size = %3d" % batch_size)
    print("loss = highly customized")
    print("optimizer = Adam 0.01")
    print("max_epochs = %3d" % max_epochs)

    print("\n Start Training")
    net.train()
    for epoch in range(0, max_epochs):
        epoch_loss = 0
        num_lines_read = 0

        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['predictors'].cuda()
            Y = batch['species'].cuda()
            optimizer.zero_grad()
            output = net(X)

            # print(output, Y)
            cel = ce_loss(output, Y)
            kll = kl_loss(net)
            tot_loss = cel + 0.1 * kll

            epoch_loss += tot_loss.item()
            tot_loss.backward()

            optimizer.step()

        if epoch % ep_log_interval == 0:
            print("epoch = %4d\tloss = %0.4f" % (epoch, epoch_loss))

    print("\nTraining Done\n")

    # Evaluation

    print("Computing Bayesian NN model accuracy")
    net.eval().cuda()
    acc = accuracy_quick(net, train_ds)
    print("Accuracy on train data = %0.4f" % acc)

    # Prediction
    print("\nPredicting species for [5.0, 2.0, 3.0, 2.0]: ")
    x = np.array([[5.0, 2.0, 3.0, 2.0]], dtype=np.float32)
    x = torch.tensor(x, dtype=torch.float32)

    for i in range(3):
        with torch.no_grad():
            logits = net(x.cuda())
        probs = torch.softmax(logits, dim=1)
        print(probs.cpu().numpy())

    print("End demo")

if __name__ == "__main__":
    main()


