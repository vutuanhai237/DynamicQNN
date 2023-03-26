import numpy as np
import classical_part
import entangled_circuit
import constant
import utilities
import time
import keras
class GradientCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        grads = self.model.optimizer.get_gradients(self.model.total_loss, self.model.trainable_weights)
        print(grads)
historiesH1 = []
test_accuraciesH1 = []
x_train, y_train, x_val, y_val, x_test, y_test = classical_part.load_mnist(
    1200, 300, 300, entangled_circuit.quanvolutional)

hmodel = classical_part.classical_model()
hmodel.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
H1 = hmodel.fit(x_train, y_train, validation_data=(x_val, y_val),
                batch_size=4, epochs=100, verbose=0, callbacks=[GradientCallback()])
historiesH1.append(H1.history)
_, test_accuracy = hmodel.evaluate(x_test, y_test)
print(test_accuracy)
