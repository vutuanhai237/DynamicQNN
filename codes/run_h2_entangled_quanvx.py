import numpy as np
import classical_part, entangled_circuit
import utilities
import multiprocessing

list_of_quanv = {
      'chain': entangled_circuit.create_Wchain_layered_ansatz,
      'alternating': entangled_circuit.create_Walternating_layered_ansatz,  
      'alltoall': entangled_circuit.create_Walltoall_layered_ansatz
}

def run_quanv(iquanv, quanv):
      historiesH2 = []
      test_accuraciesH2 = []
      for i in range(0,1):
            print('Iteration', i)
            x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test = classical_part.load_mnist_fashion(
                  1200, 300, 300, quanv, True)
            
            hmodel = classical_part.hybrid_model()
            hmodel.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
            H2 = hmodel.fit(xq_train, y_train, validation_data=(xq_val, y_val),
                  batch_size=4, epochs=100, verbose=0)
            historiesH2.append(H2.history)
            _, test_accuracy = hmodel.evaluate(xq_test, y_test)
            test_accuraciesH2.append(test_accuracy)

      utilities.save_history_train('./compare_type_quanv_fmnist/h2_4x4filter_quanv' + (iquanv), 'h2', historiesH2)
      np.savetxt('compare_type_quanv_fmnist/h2_4x4filter_quanv' + (iquanv) + '/h2test.txt', test_accuraciesH2)
      
name = 'alltoall'
run_quanv(name, list_of_quanv[name])
print("Done!")
