import numpy as np
import classical_part, entangled_circuit, constant
import utilities, time
historiesH3 = []
test_accuraciesH3 = []
   
start = time.perf_counter()
for i in range(0, 20):
      print('Iteration', i)
      x_train, y_train, x_val, y_val, x_test, y_test = classical_part.load_mnist_fashion(
            1200, 300, 300, entangled_circuit.quanvolutional)
      
      hmodel = classical_part.hybrid_model()
      hmodel.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
      H3 = hmodel.fit(x_train, y_train, validation_data=(x_val, y_val),
            batch_size=4, epochs=100, verbose=0)
      historiesH3.append(H3.history)
      _, test_accuracy = hmodel.evaluate(x_test, y_test)
      test_accuraciesH3.append(test_accuracy)
end = time.perf_counter()

utilities.save_history_train('./fmnist/h3', 'h3', historiesH3)
np.savetxt('fmnist/h3/h1test.txt', test_accuraciesH3)
np.savetxt('fmnist/h3/time.txt', [(end - start)])