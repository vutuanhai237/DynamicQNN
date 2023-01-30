import numpy as np
import classical_part, entangled_circuit
import utilities
import multiprocessing

list_of_quanv = {
      '13': entangled_circuit.quanvolutional13,
      '12': entangled_circuit.quanvolutional12,  
      '11': entangled_circuit.quanvolutional11
}

def run_quanv(iquanv, quanv):
      historiesH2 = []
      test_accuraciesH2 = []
      for i in range(0, 2):
            print('Iteration', i)
            x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test = classical_part.load_cifar10(
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

      utilities.save_history_train('./compare_type_quanv_mnist_cifar10/h2_4x4filter_quanv' + str(iquanv), 'h2', historiesH2)
      np.savetxt('compare_type_quanv_mnist_cifar10/h2_4x4filter_quanv' + str(iquanv) + '/h2test.txt', test_accuraciesH2)

threads = []

for iquanv in list_of_quanv:
      threads.append(multiprocessing.Process(target = run_quanv, args=(iquanv, list_of_quanv[iquanv])))

for thread in threads:
      thread.start()

for thread in threads:
      thread.join()

print("Done!")