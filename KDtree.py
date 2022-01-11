import math
import numpy as np
import heapq

class Node:
    def __init__(self, coordinates=None, value=None, left=None, right=None, visited=False):
        self.left = left # nó filho da esquerda
        self.right = right # nó filho da direita
        self.coordinates = coordinates # coordenadas de um ponto quando o nó for uma folha
        self.value = value # valor da mediana quando for um nó intermediário


class KDTree:
    def __init__(self, dimensions):
        self.dimensions = dimensions


    # constrói a árvore KD a partir de uma lista de coordenadas
    def build_tree(self, data, curr_dimension=0):
        if len(data) <= 1:
            return Node(coordinates=data[0])

        else:
            data = sorted(data, key=lambda x: x[curr_dimension])
            median_index = math.ceil(len(data)/2)
            temp = [x[curr_dimension] for x in data]
            median = temp[median_index]
            while(median_index < len(temp)-1 and temp[median_index] == temp[median_index+1]):
                median_index+=1
            current_node = Node(value=median)
            
            current_node.left = self.build_tree(data[:median_index], curr_dimension=((curr_dimension+1) % self.dimensions))
            current_node.right = self. build_tree(data[median_index:], curr_dimension=((curr_dimension+1) % self.dimensions))     

            self.tree = current_node
            return current_node
    

    # inicializa algumas variáveis necessárias para métodos da classe
    def initialize_search(self):
        self.heap = []
        heapq.heapify(self.heap)
        self.search_radius=float("inf")


    # calcula a distância euclidiana entre dois pontos
    def distance(self, x, y):
        x_ = np.array(x)
        y_ = np.array(y)
        return np.linalg.norm(x_-y_)


    # determina os k vizinhos mais próximos de um dado ponto
    def NN(self, point, current_node, curr_dimension=0, k=3):
        # return the k nearest neighbors of a given point
        if current_node.coordinates != None:
            # leaf
            dist = self.distance(current_node.coordinates[:-1], point[:-1])
            heapq.heappush(self.heap, (-dist, current_node.coordinates))
            if (len(self.heap) > k):
                # remove furthest neighbor
                heapq.heappop(self.heap)
            if dist < self.search_radius:
                self.search_radius = dist

        else:
            if point[curr_dimension] <= current_node.value:
                # se a mediana estiver a direita do ponto
                self.NN(point, current_node.left, curr_dimension=(curr_dimension+1)%k)
                if point[curr_dimension] + self.search_radius >= current_node.value:
                    self.NN(point, current_node.right, curr_dimension=(curr_dimension+1)%k) 

            else:
                # se a mediana estiver a esquerda do ponto
                self.NN(point, current_node.right, curr_dimension=(curr_dimension+1)%k) 
                if point[curr_dimension] - self.search_radius <= current_node.value:
                    self.NN(point, current_node.left, curr_dimension=(curr_dimension+1)%k)