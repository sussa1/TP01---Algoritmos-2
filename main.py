from scipy.sparse import data
from KNN import x_NN
from data import Data
import sys


if(len(sys.argv) > 1):
    files = []
    for i in range(1, len(sys.argv)):
        files.append(sys.argv[i])
else:
    files=["banana.dat"]


def run(files):
    for file in files:
        # lê a base de dados e separa em conjunto de teste e treino
        data_frame = Data(file)
        data_frame.generate_train_test_sets()

        print(len(data_frame.data_list))

        # classifica os pontos do conjunto de treino
        kNN = x_NN(data_frame.test_set, data_frame.training_set, k=5)
        y_pred = kNN.kNN()

        y_real = [y[-1] for y in data_frame.test_set]

        # obtém as estatísticas
        cm, accuracy, precision, recall = kNN.statistics(y_pred, y_real)

        print("\""+file+"\"")
        print("   accuracy = " + str(accuracy))
        print("   precision = " + str(precision))
        print("   recall = " + str(recall))
        print()

run(files)