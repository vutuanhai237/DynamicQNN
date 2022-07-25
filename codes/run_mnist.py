import numpy as np
import classical_part
import utilities
historiesH1 = []
test_accuraciesH1 = []
historiesH2 = []
test_accuraciesH2 = []
for i in range(0, 20):
    x_train, xq_train, y_train, x_val, xq_val, y_val, x_test, xq_test, y_test = classical_part.load_mnist(1200, 400, 400)
    
    
    cmodel = classical_part.classical_model()
    cmodel.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    H1 = cmodel.fit(x_train, y_train, validation_data=(x_val, y_val),
            batch_size=4, epochs=30, verbose=0)
    
    historiesH1.append(H1.history)
    _, test_accuracy = cmodel.evaluate(x_test, y_test)
    test_accuraciesH1.append(test_accuracy)

    
    hmodel = classical_part.hybrid_model()
    hmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    H2 = hmodel.fit(xq_train, y_train, validation_data=(xq_val, y_val),
          batch_size=4, epochs=30, verbose=0)
    historiesH2.append(H2.history)
    _, test_accuracy = hmodel.evaluate(xq_test, y_test)
    test_accuraciesH2.append(test_accuracy)
    

    hmodel1 = classical_part.hybrid_model()
    hmodel1.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    H3 = hmodel1.fit(x_train, y_train, validation_data=(x_val, y_val),
          batch_size=4, epochs=30, verbose=0)
    historiesH3 = []
    test_accuraciesH3 = []
    historiesH3.append(H3.history)
    _, test_accuracy = hmodel1.evaluate(x_test, y_test)
    test_accuraciesH3.append(test_accuracy)

utilities.save_history_train('./exps_mnist', 'h1', historiesH1)
np.savetxt('exps_mnist/h1test.txt', test_accuraciesH1)
utilities.save_history_train('./exps_mnist', 'h2', historiesH2)
np.savetxt('exps_mnist/h2test.txt', test_accuraciesH2)
utilities.save_history_train('./exps_mnist', 'h3', historiesH3)
np.savetxt('exps_mnist/h3test.txt', test_accuraciesH3)
