import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import time

class SpMatrix:
    def __init__(self, diag, size, shift) -> None:
        self.size = size
        self.matrix = np.zeros((size, size))

        for i in range(size):
            self.matrix[i, i] = diag

            if self._validIndecies(i, i - 1):
                self.matrix[i, i - 1] = -1
            if self._validIndecies(i, i + 1):
                self.matrix[i, i + 1] = -1
            
            if self._validIndecies(i - shift, i):
                self.matrix[i - shift, i] = -1
            if self._validIndecies(i + shift, i):
                self.matrix[i + shift, i] = -1

    def _validIndecies(self, i, j):
        if (i >= self.size or i < 0 or j >= self.size or j < 0):
            return False
        return True
    
    def decompositionLU(self):
        P, L, U = scipy.linalg.lu(self.matrix)
        return L, U

    
class diffSLAE:
    def __init__(self, omegaSize, h, c) -> None:
        
        self.numStrokes = (int)(omegaSize/h) + 1
        if self.numStrokes < 2:
            print("h must be much larger than omegaSize")
            return
        
        self.h = h
        self.c = c
        self.numNodes = self.numStrokes * self.numStrokes

        diag = 4 + c*h*h

        self.L = SpMatrix(diag, self.numNodes, self.numStrokes)

    def saveMyL(self, name) -> None:
        plt.figure(figsize=[12,9], dpi=120)
        
        plt.spy(self.L.matrix, markersize=1, marker='.', color='blue')

        plt.xticks(np.arange(0, self.numNodes, step=10))
        plt.yticks(np.arange(0, self.numNodes, step=10))
        plt.grid()
        
        plt.title("L Matrix of SLAE")

        plt.savefig(name)

        plt.close()

    def solveMe(self):
        b = np.zeros(self.numNodes)
        b += self.h*self.h

        L, U = self.L.decompositionLU()

        y = scipy.linalg.solve_triangular(L, b, lower=True)
        x = scipy.linalg.solve_triangular(U, y)

        return x
    
    def saveMySolution(self, name):
        norm = self.h
        solutionDiffSLAE(self.solveMe(), self.numStrokes, norm, norm).saveMe(name)

class solutionDiffSLAE:
    def __init__(self, x, n, normX, normY) -> None:
        if n*n != x.size:
            print('smth wrong, i can feel it!')

        self.values = np.zeros((n, n))

        self.x = np.arange(n)*normX
        self.y = np.arange(n)*normY

        for i in range(n):
            for j in range(n):
                k = 5*i + j
                self.values[i, j] = x[k]
        
    def saveMe(self, name):
        plt.figure(figsize=[12,9], dpi=120)
        
        plt.pcolormesh(self.x, self.y, self.values, cmap='viridis', shading='auto')
        plt.colorbar()
        plt.title("solution of DE")

        plt.savefig(name)

        plt.close()

#####################################################################################
#.................................MAIN..............................................#
#####################################################################################

omegaSize = 1
#h = 0.01
c = 0.1 

#diffSLAE(omegaSize, h, c).saveMySolution("solutionHquater")

h_values = np.array([0.01, 0.025, 0.05, 0.0625, 0.1, 0.125, 0.20, 0.25])
time_values = np.zeros(h_values.size)

def shotAnExperiment(h):
    start = time.time()
    diffSLAE(omegaSize, h, c).solveMe()
    end = time.time()

    return end - start

for i, h in enumerate(h_values):
    sumTime = 0

    for j in range(10):
        sumTime += shotAnExperiment(h)
    
    time_values[i] = sumTime/10 

# Создайте DataFrame из массивов
df = pd.DataFrame({'h': h_values, 'execTime': time_values})

# Сохраните DataFrame в CSV файл
df.to_csv('Time(h).csv', index=False)  # 'experiment_results.csv' - имя файла

plt.figure(figsize=[12,9], dpi=120)

plt.plot(h_values, time_values)

plt.xscale('log', base = 2)
plt.yscale('log')

plt.xlabel("$h$ в $log_2$ масштабе")
plt.ylabel("Среднее время вычисления в $log_10$ масштабе")

plt.grid()

plt.title("Зависимость времени решения от $h$")

plt.savefig("Time(h)")

plt.close()
