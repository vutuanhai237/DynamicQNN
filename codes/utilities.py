import numpy as np
def save_history_train(path, model_name, histories):
    losses = []
    val_losses = []
    accuracies = []
    val_accuracies = []
    for history in histories:
        losses.append(history['loss'])
        val_losses.append(history['val_loss'])
        accuracies.append(history['accuracy'])
        val_accuracies.append(history['val_accuracy'])
    np.savetxt(path + '/' + model_name + 'history_loss.txt', np.asarray(losses))
    np.savetxt(path + '/' + model_name + 'history_val_loss.txt', np.asarray( val_losses))
    np.savetxt(path + '/' + model_name + 'history_accuracy.txt',  np.asarray(accuracies))
    np.savetxt(path + '/' + model_name + 'history_val_accuracy.txt',  np.asarray(val_accuracies))
    return
