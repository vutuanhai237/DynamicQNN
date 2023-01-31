import numpy as np
import classical_part, entangled_circuit, constant
import utilities, time
historiesH2 = []
test_accuraciesH2 = []
   
start = time.perf_counter()
for i in range(0, 10):
      print('Iteration', i)
      x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test = classical_part.load_cifar10(
            1200, 300, 300, entangled_circuit.quanvolutional, True)
      
      hmodel = classical_part.hybrid_model()
      hmodel.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
      H2 = hmodel.fit(xq_train, y_train, validation_data=(xq_val, y_val),
            batch_size=4, epochs=100, verbose=0)
      historiesH2.append(H2.history)
      _, test_accuracy = hmodel.evaluate(xq_test, y_test)
      test_accuraciesH2.append(test_accuracy)
end = time.perf_counter()

utilities.save_history_train('./cifar10/h2_' + str(constant.quanv_num_filter) + 'filter' + str(
    constant.quanv_size_filter) + 'x' + str(constant.quanv_size_filter), 'h2', historiesH2)
np.savetxt('cifar10/h2_' + str(constant.quanv_num_filter) + 'filter' + str(constant.quanv_size_filter) + 'x' + str(constant.quanv_size_filter) + '/h2test.txt', test_accuraciesH2)
np.savetxt('cifar10/h2_' + str(constant.quanv_num_filter) + 'filter' + str(constant.quanv_size_filter) + 'x' + str(constant.quanv_size_filter) + '/time.txt', [(end - start)])
