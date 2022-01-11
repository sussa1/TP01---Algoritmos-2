from KDtree import KDTree as KDT
from sklearn.metrics import confusion_matrix

def mode(l):
    return max(tuple(l), key = l.count)

class x_NN:
    def __init__(self, test, train, k=3):
        self.test = test
        self.train = train
        self.k = k

    def kNN(self):
        # constrói a árvore KD para o conjunto de teste
        KD_tree = KDT(dimensions=len(self.train[0])-1)
        KD_tree.build_tree(self.train)
        prediction = []

        # para cada ponto no conjunto de teste classifica-o utilizando kNN 
        for point in self.test:
            KD_tree.initialize_search()
            KD_tree.NN(point, KD_tree.tree, k=self.k)
            nearest_neighbors = KD_tree.heap
            nearest_neighbors_class = [x[1][-1] for x in nearest_neighbors]
            classification = mode(nearest_neighbors_class)
            prediction.append(classification)

        # retorna uma lista com o valor predito para classe de todas as instâncioas de teste
        return prediction

    def statistics(self, y_pred, y_real):
        cm = confusion_matrix(y_real, y_pred)

        #                 | Predicted Positive  |  Predicted Negative |
        # Actual Positive |    True Positive    |   False Positive    |
        # Actual Negative |   False Negative    |    True Negative    |

        TP = cm[0][0]
        FN = cm[1][0]
        FP = cm[0][1]
        TN = cm[1][1]

        accuracy = (TP+TN)/(TP+FP+FN+TN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        
        return cm, accuracy, precision, recall