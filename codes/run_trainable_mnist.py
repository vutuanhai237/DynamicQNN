import numpy as np
import classical_part
import utilities
historiesH1 = []
test_accuraciesH1 = []
historiesH2 = []
test_accuraciesH2 = []
historiesH3 = []
test_accuraciesH3 = []
for i in range(0, 20):
    print('Iteration', i)
    x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test = classical_part.load_mnist(
        1200, 400, 400)

    hmodel = classical_part.hybrid_model()
    hmodel.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
    for i in range(0,30):

        H2 = hmodel.fit(xq_train, y_train, validation_data=(xq_val, y_val),
                        batch_size=4, epochs=30, verbose=0)
    historiesH2.append(H2.history)
    _, test_accuracy = hmodel.evaluate(xq_test, y_test)
    test_accuraciesH2.append(test_accuracy)


utilities.save_history_train('./exps_mnist', 'h2', historiesH2)
np.savetxt('exps_mnist/h2test.txt', test_accuraciesH2)
