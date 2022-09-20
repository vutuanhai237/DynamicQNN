import numpy as np
import classical_part, entangled_circuit, constant
import utilities, time
historiesH1 = []
test_accuraciesH1 = []
   
start = time.perf_counter()
for i in range(0, 20):
      print('Iteration', i)
      x_train, y_train, x_val, y_val, x_test, y_test = classical_part.load_mnist(
            1200, 300, 300, entangled_circuit.quanvolutional, False)
      
      hmodel = classical_part.hybrid_model()
      hmodel.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
      H1 = hmodel.fit(x_train, y_train, validation_data=(x_val, y_val),
            batch_size=4, epochs=100, verbose=0)
      historiesH1.append(H1.history)
      _, test_accuracy = hmodel.evaluate(x_test, y_test)
      test_accuraciesH1.append(test_accuracy)
end = time.perf_counter()

utilities.save_history_train('./exps_mnist/h1_' + str(constant.num_conv_filter) + 'filter', 'h2', historiesH2)
np.savetxt('exps_mnist/h1_' + str(constant.num_conv_filter) + 'filter/h2test.txt', test_accuraciesH2)
np.savetxt('exps_mnist/h1_' + str(constant.num_conv_filter) + 'filter/time.txt', [(end - start)])