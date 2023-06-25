import numpy as np 
import matplotlib.pyplot as plt

def plot_acc_loss(results):
    """
    Training data statistics
    """
    x = np.array([i for i in range(len(results["train_acc"]))])

    plt.plot(x, results["train_acc"], color='b', label="Train acc")
    plt.plot(x, results["train_loss"], color='r', label="Train loss")
    plt.plot(x, results["test_acc"], color='g', label="Test acc")
    plt.plot(x, results["test_loss"], color='m', label="Test loss")

    plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    plt.title("Statistics")

    plt.legend()

    print(f"X: {len(x)}")

    plt.xlim([0, len(x)])
    plt.ylim([0, 100.0])

    plt.show()
