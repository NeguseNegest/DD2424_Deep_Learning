import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import tarfile
import urllib.request
import os
import shutil
import torch
import copy
from tqdm import tqdm


from torch_gradient_computations import ComputeGradsWithTorch




def LoadBatch(filename):
    with open(filename, "rb") as fo:
        batch = pickle.load(fo, encoding="bytes")

    X = batch[b"data"].astype(np.float64).T / 255.0   # 3072 x 10000  divide by 255.0 so to keep everything small 
    y = np.array(batch[b"labels"])                    # length 10000, since we have 1000 images

    K = 10
    n = X.shape[1]
    Y = np.zeros((K, n), dtype=X.dtype)
    Y[y, np.arange(n)] = 1 # each column represent the class representation of one image 

    return X, Y, y

def LoadAllTrainingData(filepath, val_size=1000, rng=None):
    X_list = []
    Y_list = []
    y_list = []

    for i in range(1, 6):
        X, Y, y = LoadBatch(filepath + f"data_batch_{i}")
        X_list.append(X)
        Y_list.append(Y)
        y_list.append(y)

    X_all = np.hstack(X_list)          # (3072, 50000)
    Y_all = np.hstack(Y_list)          # (10, 50000)
    y_all = np.concatenate(y_list)     # (50000,)

    n = X_all.shape[1]

    if rng is None:
        rng = np.random.default_rng(42)

    perm = rng.permutation(n)
    X_all = X_all[:, perm]
    Y_all = Y_all[:, perm]
    y_all = y_all[perm]

    X_val = X_all[:, :val_size]
    Y_val = Y_all[:, :val_size]
    y_val = y_all[:val_size]

    X_train = X_all[:, val_size:]
    Y_train = Y_all[:, val_size:]
    y_train = y_all[val_size:]

    return X_train, Y_train, y_train, X_val, Y_val, y_val


def normalize(X, mean_X, std_X):
    return (X - mean_X) / std_X

def softmax(s):
    s = s - np.max(s, axis=0, keepdims=True)   # stability
    exp_s = np.exp(s)
    return exp_s / np.sum(exp_s, axis=0, keepdims=True)



def init_network(X_train):

    K = 10
    d = X_train.shape[0]

    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    seed = 42
    rng.bit_generator.state = BitGen(seed).state

    init_net = {
        'W': 0.01 * rng.standard_normal((K, d)),
        'b': np.zeros((K, 1))
    }
    return init_net

def ApplyNetwork(X, network):
    W = network['W']
    b = network['b']

    s = W @ X + b
    P = softmax(s)
    return P


def ComputeLoss(P, y):
    n = P.shape[1]
    correct_class_probs = P[y, np.arange(n)]
    L = -np.mean(np.log(correct_class_probs))
    return L

def BackwardPass(X, Y, P, network, lam):
    W = network['W']
    n = X.shape[1]

    G = P - Y

    grads = {}
    grads['W'] = (G @ X.T) / n + 2 * lam * W
    grads['b'] = np.sum(G, axis=1, keepdims=True) / n

    return grads

def computeAccuracy(P, y):
    y_pred = np.argmax(P, axis=0)
    accuracy = np.mean(y_pred == y)
    return 100*accuracy

def ComputeCost(X, y, network, lam):
    P = ApplyNetwork(X, network)
    loss = ComputeLoss(P, y)
    cost = loss + lam * np.sum(network['W'] ** 2)
    return cost, loss



def MiniBatchGD(X_train, Y_train, y_train, X_val, y_val, GDparams, init_net, lam,
                rng=None, flip_prob=0.0, inds_flip=None, step_decay_every=None, step_decay_factor=0.1):

    n_batch = GDparams['n_batch']
    eta = GDparams['eta']
    n_epochs = GDparams['n_epochs']

    trained_net = copy.deepcopy(init_net)

    n = X_train.shape[1]

    train_costs = []
    train_losses = []
    val_costs = []
    val_losses = []

    if rng is None:
        rng = np.random.default_rng(42)

    current_eta = eta

    for epoch in range(n_epochs):
        perm = rng.permutation(n)

        X_train_epoch = X_train[:, perm]
        Y_train_epoch = Y_train[:, perm]
        y_train_epoch = y_train[perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch

            Xbatch = X_train_epoch[:, j_start:j_end]
            Ybatch = Y_train_epoch[:, j_start:j_end]

            if flip_prob > 0.0 and inds_flip is not None:
                Xbatch = flip_batch_horizontally(Xbatch, inds_flip, rng, p=flip_prob)

            Pbatch = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, Pbatch, trained_net, lam)

            trained_net['W'] -= current_eta * grads['W']
            trained_net['b'] -= current_eta * grads['b']

        train_cost, train_loss = ComputeCost(X_train, y_train, trained_net, lam)
        val_cost, val_loss = ComputeCost(X_val, y_val, trained_net, lam)

        train_costs.append(train_cost)
        train_losses.append(train_loss)
        val_costs.append(val_cost)
        val_losses.append(val_loss)

        if step_decay_every is not None and (epoch + 1) % step_decay_every == 0:
            current_eta *= step_decay_factor

    history = {
        'train_costs': train_costs,
        'train_losses': train_losses,
        'val_costs': val_costs,
        'val_losses': val_losses
    }

    return trained_net, history


def plot_history(history, eta=0.001, lam=0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['train_costs'], label=f'Train eta={eta}, lam={lam}', color='forestgreen', lw=2)
    ax1.plot(history['val_costs'], label=f'Val eta={eta}, lam={lam}', color='crimson', linestyle='--')
    ax1.set_title('Model Cost over Epochs', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cost')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_losses'], label=f'Train eta={eta}, lam={lam}', color='forestgreen', lw=2)
    ax2.plot(history['val_losses'], label=f'Val eta={eta}, lam={lam}', color='crimson', linestyle='--')
    ax2.set_title('Model Loss over Epochs', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_all_histories(history_experiments):
    n_exp = len(history_experiments)
    fig, axes = plt.subplots(2, n_exp, figsize=(5 * n_exp, 8), squeeze=False)

    for col, ((eta, lam), history) in enumerate(history_experiments.items()):
        # Cost
        axes[0, col].plot(history['train_costs'], label='Train', color='forestgreen', lw=2)
        axes[0, col].plot(history['val_costs'], label='Val', color='crimson', linestyle='--')
        axes[0, col].set_title(f'Cost\neta={eta}, lam={lam}')
        axes[0, col].set_xlabel('Epoch')
        axes[0, col].set_ylabel('Cost')
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend()

        # Loss
        axes[1, col].plot(history['train_losses'], label='Train', color='forestgreen', lw=2)
        axes[1, col].plot(history['val_losses'], label='Val', color='crimson', linestyle='--')
        axes[1, col].set_title(f'Loss\neta={eta}, lam={lam}')
        axes[1, col].set_xlabel('Epoch')
        axes[1, col].set_ylabel('Loss')
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend()

    plt.tight_layout()
    plt.show()

def EvaluateSet(X, y, network, lam):
    P = ApplyNetwork(X, network)
    acc = computeAccuracy(P, y)
    loss = ComputeLoss(P, y)
    cost = loss + lam * np.sum(network['W']**2)
    return {"acc": acc, "loss": loss, "cost": cost}



def plot_all_weight_images(results, class_names=None, figsize_per_cell=(1.8, 1.8)):
    if class_names is None:
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    n_exp = len(results)
    n_classes = 10

    fig, axs = plt.subplots(
        n_exp, n_classes,
        figsize=(figsize_per_cell[0] * n_classes, figsize_per_cell[1] * n_exp),
        squeeze=False
    )

    for row, res in enumerate(results):
        W = res['trained_net']['W']   # (10, 3072)

        Ws = W.T.reshape((32, 32, 3, 10), order='F')
        W_im = np.transpose(Ws, (1, 0, 2, 3))

        eta = res['exp']['eta']
        lam = res['exp']['lam']

        for col in range(n_classes):
            ax = axs[row, col]
            w_im = W_im[:, :, :, col]

            w_min = np.min(w_im)
            w_max = np.max(w_im)
            w_im_norm = (w_im - w_min) / (w_max - w_min + 1e-12)

            ax.imshow(w_im_norm)
            ax.axis("off")

            if row == 0:
                ax.set_title(class_names[col], fontsize=10)

            if col == 0:
                ax.set_ylabel(f"eta={eta}\nlam={lam}", fontsize=10)

    plt.tight_layout()
    plt.show()


def run_experiments():
    experiments = [
        {"lam": 0.0, "eta": 0.1,   "n_batch": 100, "n_epochs": 40},
        {"lam": 0.0, "eta": 0.001, "n_batch": 100, "n_epochs": 40},
        {"lam": 0.1, "eta": 0.001, "n_batch": 100, "n_epochs": 40},
        {"lam": 1.0, "eta": 0.001, "n_batch": 100, "n_epochs": 40},
    ]

    results = []
    history_eta_lam = {}

    for exp in experiments:
        init_net = init_network(X_train)

        GDparams = {
            "n_batch": exp["n_batch"],
            "eta": exp["eta"],
            "n_epochs": exp["n_epochs"]
        }

        trained_net, history = MiniBatchGD(
            X_train, Y_train, y_train,
            X_validation, y_validation,
            GDparams, init_net, exp["lam"], rng
        )

        key = (exp["eta"], exp["lam"])   # hashable
        history_eta_lam[key] = history

        test_metrics = EvaluateSet(X_test, y_test, trained_net, exp["lam"])

        results.append({
            "exp": exp,
            "trained_net": trained_net,
            "history": history,
            "test_acc": test_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_cost": test_metrics["cost"]
        })

        print(exp, "-> test acc:", test_metrics["acc"])
    
    return results, history_eta_lam

def make_flip_indices():
    aa = np.int32(np.arange(32)).reshape((32, 1))
    bb = np.int32(np.arange(31, -1, -1)).reshape((32, 1))
    vv = np.tile(32 * aa, (1, 32))
    ind_flip = vv.reshape((32 * 32, 1)) + np.tile(bb, (32, 1))
    inds_flip = np.vstack((ind_flip, 1024 + ind_flip))
    inds_flip = np.vstack((inds_flip, 2048 + ind_flip))
    return inds_flip.flatten()

def flip_batch_horizontally(Xbatch, inds_flip, rng, p=0.5):
    X_aug = Xbatch.copy()
    n = Xbatch.shape[1]

    flip_mask = rng.random(n) < p
    X_aug[:, flip_mask] = X_aug[inds_flip, :][:, flip_mask]

    return X_aug

def bonus_grid_search(X_train, Y_train, y_train, X_val, y_val, X_test, y_test):
    inds_flip = make_flip_indices()

    grid = {
        "lam": [0.0, 1e-5, 1e-4, 1e-3],
        "eta": [0.001, 0.002, 0.003, 0.004],
        "n_batch": [100, 200]
    }

    best = None
    all_results = []

    n_train = X_train.shape[1]

    # Fair comparison target: same number of updates as 40 epochs with batch size 100
    ref_batch = 100
    ref_epochs = 40
    target_updates = ref_epochs * (n_train // ref_batch)

    for lam in tqdm(grid["lam"]):
        for eta in tqdm(grid["eta"]):
            for n_batch in tqdm(grid["n_batch"]):

                updates_per_epoch = n_train // n_batch
                n_epochs = math.ceil(target_updates / updates_per_epoch)

                GDparams = {
                    "n_batch": n_batch,
                    "eta": eta,
                    "n_epochs": n_epochs
                }

                init_net = init_network(X_train)

                # fresh RNG for this run -> fairer and reproducible
                run_rng = np.random.default_rng(42)

                trained_net, history = MiniBatchGD(
                    X_train, Y_train, y_train,
                    X_val, y_val,
                    GDparams, init_net, lam,
                    rng=run_rng,
                    flip_prob=0.5,
                    inds_flip=inds_flip,
                    step_decay_every=None,
                    step_decay_factor=0.1
                )

                val_metrics = EvaluateSet(X_val, y_val, trained_net, lam)

                result = {
                    "lam": lam,
                    "eta": eta,
                    "n_batch": n_batch,
                    "n_epochs": n_epochs,
                    "val_acc": val_metrics["acc"],
                    "trained_net": trained_net,
                    "history": history
                }

                all_results.append(result)

                if best is None or result["val_acc"] > best["val_acc"]:
                    best = result

                print(
                    f"lam={lam}, eta={eta}, n_batch={n_batch}, "
                    f"n_epochs={n_epochs}, val_acc={val_metrics['acc']:.2f}"
                )

    best_test_metrics = EvaluateSet(X_test, y_test, best["trained_net"], best["lam"])
    best["test_acc"] = best_test_metrics["acc"]
    best["test_loss"] = best_test_metrics["loss"]
    best["test_cost"] = best_test_metrics["cost"]

    return best, all_results

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    FILEPATH="Assignment_1/Datasets/cifar-10-batches-py/" # we add the directory path first 

    # X_train,Y_train,y_train=LoadBatch(FILEPATH+"data_batch_1")

    # X_validation,Y_validation,y_validation=LoadBatch(FILEPATH+"data_batch_2")

    # X_test,Y_test,y_test=LoadBatch(FILEPATH+"test_batch")

    # d = X_train.shape[0]

    # mean_X = np.mean(X_train, axis=1).reshape(d, 1)
    # std_X  = np.std(X_train, axis=1).reshape(d, 1)

    # X_train = normalize(X_train, mean_X, std_X)
    # X_validation = normalize(X_validation, mean_X, std_X)
    # X_test = normalize(X_test, mean_X, std_X)


    # rng = np.random.default_rng()
    # BitGen = type(rng.bit_generator)
    # seed = 42
    # rng.bit_generator.state = BitGen(seed).state

    # d_small = 10
    # n_small = 3
    # lam = 0.0

    # small_net = {}
    # small_net['W'] = 0.01 * rng.standard_normal(size=(10, d_small))
    # small_net['b'] = np.zeros((10, 1))

    # X_small = X_train[0:d_small, 0:n_small]
    # Y_small = Y_train[:, 0:n_small]
    # y_small = y_train[0:n_small]

    # P_small = ApplyNetwork(X_small, small_net)

    # my_grads = BackwardPass(X_small, Y_small, P_small, small_net, lam)
    # torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net)

    # print("max abs diff W:", np.max(np.abs(my_grads['W'] - torch_grads['W'])))
    # print("max abs diff b:", np.max(np.abs(my_grads['b'] - torch_grads['b'])))

    # def relative_error(a, b, eps=1e-10):
    #     return np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b))

    # rel_W = relative_error(my_grads['W'], torch_grads['W'])
    # rel_b = relative_error(my_grads['b'], torch_grads['b'])

    # print("max relative error W:", np.max(rel_W))
    # print("max relative error b:", np.max(rel_b))

    # lam = 0.1

    # P_small = ApplyNetwork(X_small, small_net)
    # my_grads = BackwardPass(X_small, Y_small, P_small, small_net, lam)
    # torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net, lam)

    # print("max abs diff W:", np.max(np.abs(my_grads['W'] - torch_grads['W'])))
    # print("max abs diff b:", np.max(np.abs(my_grads['b'] - torch_grads['b'])))
    # print("max relative error W:", np.max(relative_error(my_grads['W'], torch_grads['W'])))
    # print("max relative error b:", np.max(relative_error(my_grads['b'], torch_grads['b'])))


    # GDparams = {
    # 'n_batch': 100,
    # 'eta': 0.001,
    # 'n_epochs': 40
    # }


    # network=init_network(X_train)

    # trained_net, history = MiniBatchGD(
    #     X_train, Y_train, y_train,
    #     X_validation, y_validation,
    #     GDparams, network, lam=0.0, rng=rng
    # )

    # plot_history(history)



    # test_metrics = EvaluateSet(X_test, y_test, trained_net, lam=0)
    # print("Final test accuracy:", test_metrics["acc"])
    # print("Final test loss:", test_metrics["loss"])
    # print("Final test cost:", test_metrics["cost"])


    # results, history_experiments = run_experiments()

    # plot_all_weight_images(results)

    # plot_all_histories(history_experiments)

#------------BONUS POINT EXERCISE 2 START-----------------------#

    #----------- 2.a--------------------#


    X_train, Y_train, y_train, X_validation, Y_validation, y_validation = LoadAllTrainingData(FILEPATH, val_size=1000, rng=rng)
    X_test, Y_test, y_test = LoadBatch(FILEPATH + "test_batch")

    d = X_train.shape[0]
    mean_X = np.mean(X_train, axis=1).reshape(d, 1)
    std_X = np.std(X_train, axis=1).reshape(d, 1) + 1e-12

    X_train = normalize(X_train, mean_X, std_X)
    X_validation = normalize(X_validation, mean_X, std_X)
    X_test = normalize(X_test, mean_X, std_X)


        # baseline with all 5 batches, no augmentation
    baseline_bonus_params = {
        "n_batch": 100,
        "eta": 0.001,
        "n_epochs": 40
    }

    baseline_bonus_net = init_network(X_train)
    baseline_bonus_trained, baseline_bonus_history = MiniBatchGD(
        X_train, Y_train, y_train,
        X_validation, y_validation,
        baseline_bonus_params,
        baseline_bonus_net,
        lam=0.0,
        rng=rng,
        flip_prob=0.0,
        inds_flip=None,
        step_decay_every=None
    )

    baseline_bonus_val_metrics = EvaluateSet(X_validation, y_validation, baseline_bonus_trained, lam=0.0)

    print("\nBonus baseline:")
    print("val acc :", baseline_bonus_val_metrics["acc"])

  
    # full bonus search
    best_bonus, all_bonus_results = bonus_grid_search(
        X_train, Y_train, y_train,
        X_validation, y_validation,
        X_test, y_test,
    )

    print("\nBest bonus result:")
    print("lambda   :", best_bonus["lam"])
    print("eta      :", best_bonus["eta"])
    print("n_batch  :", best_bonus["n_batch"])
    print("val acc  :", best_bonus["val_acc"])
    print("test acc :", best_bonus["test_acc"])
    
    baseline_bonus_test_metrics = EvaluateSet(X_test, y_test, baseline_bonus_trained, lam=0.0)

    print("\nFinal comparison on test set:")
    print("baseline test acc :", baseline_bonus_test_metrics["acc"])
    print("best bonus test acc :", best_bonus["test_acc"])


    plot_history(
        best_bonus["history"],
        eta=best_bonus["eta"],
        lam=best_bonus["lam"]
    )
