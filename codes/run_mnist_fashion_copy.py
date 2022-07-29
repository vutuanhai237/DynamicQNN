import numpy as np
import classical_part
import utilities

historiesH3 = []
test_accuraciesH3 = []
for i in range(0, 20):
      print('Iteration', i)
      x_train, y_train, x_val, y_val, x_test, y_test = classical_part.load_mnist_fashion(1200, 400, 400)
      hmodel1 = classical_part.hybrid_model()
      hmodel1.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
      H3 = hmodel1.fit(x_train, y_train, validation_data=(x_val, y_val),
                     batch_size=4, epochs=30, verbose=0)

      historiesH3.append(H3.history)
      _, test_accuracy = hmodel1.evaluate(x_test, y_test)
      test_accuraciesH3.append(test_accuracy)
    

utilities.save_history_train('./exps_mnist_fashion/h3', 'h3', historiesH3)
np.savetxt('exps_mnist_fashion/h3/h3test.txt', test_accuraciesH3)
