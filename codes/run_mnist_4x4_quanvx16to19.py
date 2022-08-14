import numpy as np
import classical_part, entangled_circuit
import utilities

historiesH2 = []
test_accuraciesH2 = []
list_of_quanv = {
      '16': entangled_circuit.quanvolutional16, 
      '17': entangled_circuit.quanvolutional17,
      '18': entangled_circuit.quanvolutional18,
      '19': entangled_circuit.quanvolutional19
}

for iquanv in list_of_quanv:
      for i in range(0, 20):
            print('Iteration', i)

            x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test = classical_part.load_mnist(
                  1200, 300, 300, list_of_quanv[iquanv], True)
            
            hmodel = classical_part.hybrid_model()
            hmodel.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
            H2 = hmodel.fit(xq_train, y_train, validation_data=(xq_val, y_val),
                  batch_size=4, epochs=100, verbose=0)
            historiesH2.append(H2.history)
            _, test_accuracy = hmodel.evaluate(xq_test, y_test)
            test_accuraciesH2.append(test_accuracy)
      

      utilities.save_history_train('./exps_mnist/h2_4x4filter_quanv' + str(iquanv), 'h2', historiesH2)
      np.savetxt('exps_mnist/h2_4x4filter_quanv' + str(iquanv) + '/h2test.txt', test_accuraciesH2)
