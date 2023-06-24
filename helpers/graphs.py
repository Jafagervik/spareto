import numpy as np 
import matplotlib.pyplot as plt

def plot_acc_loss(results):
    """
    Training data statistics
    """
    train_a, train_l, test_a, test_l = results 

    x = np.arange(0, len(train_a), 1)

    plt.plot(x, train_a, color='o', label='Train acc')
    plt.plot(x, train_l, color='r', label='Train loss')
    plt.plot(x, test_a, color='g', label='Test acc')
    plt.plot(x, test_l, color='p', label='Test loss')

    plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    plt.title("Statistics")

    plt.legend()

    plt.xlim([0, len(x)])
    plt.ylim([0, 100.0])

    plt.show()
