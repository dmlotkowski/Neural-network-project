import matplotlib.pyplot as plt
import tensorflow as tf
from model import hist

# Recreate the exact same model, including its weights and the optimizer



#Plot loss and accuracy for the training and validation set.
def plot_hist(hist):
    loss_list = [s for s in hist.hist.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in hist.hist.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in hist.hist.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in hist.hist.keys() if 'acc' in s and 'val' in s]
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    plt.figure(figsize=(22,10))
    ## As loss always exists
    epochs = range(1,len(hist.hist[loss_list[0]]) + 1)
    ## Accuracy
    plt.figure(221, figsize=(20,10))
    ## Accuracy
    # plt.figure(2,figsize=(14,5))
    plt.subplot(221, title='Accuracy')
    for l in acc_list:
        plt.plot(epochs, hist.hist[l], 'b', label='Training accuracy (' + str(format(hist.hist[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, hist.hist[l], 'g', label='Validation accuracy (' + str(format(hist.hist[l][-1],'.5f'))+')')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    ## Loss
    plt.subplot(222, title='Loss')
    for l in loss_list:
        plt.plot(epochs, hist.hist[l], 'b', label='Training loss (' + str(str(format(hist.hist[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, hist.hist[l], 'g', label='Validation loss (' + str(str(format(hist.hist[l][-1],'.5f'))+')'))    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # plot history
    plot_hist(hist)