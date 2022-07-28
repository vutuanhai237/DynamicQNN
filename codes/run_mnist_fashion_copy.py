import numpy as np
import classical_part
import utilities

historiesH1 = []
test_accuraciesH1 = []
for i in range(0, 20):
      print('Iteration', i)
      x_train, y_train, x_val, y_val, x_test, y_test = classical_part.load_mnist_fashion(1200, 400, 400)
      
      cmodel = classical_part.classical_model()
      cmodel.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
      H1 = cmodel.fit(x_train, y_train, validation_data=(x_val, y_val),
                  batch_size=4, epochs=30, verbose=0)
      
      historiesH1.append(H1.history)
      _, test_accuracy = cmodel.evaluate(x_test, y_test)
      test_accuraciesH1.append(test_accuracy)
    

utilities.save_history_train('./exps_mnist_fashion/h1_50filter', 'h1', historiesH1)
np.savetxt('exps_mnist_fashion/h1_50filter/h1test.txt', test_accuraciesH1)
