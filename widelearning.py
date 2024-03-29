
import numpy as np
import pandas as pd
import os
import shutil
import glob
import math as m
import requests
import csv

def txt_kernel(file_path):	
    '''
    Функция преобразования сгенерированного ядра для вставки в структуру TensorFlow 
    (для черно-белых, одноканальных изображений)

    Параметры
    ----------
    file_path : str
        путь к текстовому файлу, содержащему сверточное ядро, записанное в следующем виде 
        (с запятыми в конце каждой строки):
        1,2,3,
        4,5,6,
        7,8,9,

    Пример использования:
    wdl.txt_kernel('horizontal.txt')

    '''
    f = open(file_path, 'r')
	
    weights = []

    for i in f:
        weights.append(i.strip().split(','))

    for j in weights:
        j.pop()

    for ii in weights:
        for i1, elem in enumerate(ii):
            ii[i1] = [[int(elem)]]
            
    with open('kernel.txt', 'w') as file:
        file.write(str(weights))

    return weights 

def add_kernel(path, init_kernel):
    '''
    Функция позволяет устанавливать дополнительные ядра сверточного слоя к уже установленным исходным
    (для черно-белых, одноканальных изображений)

    Параметры
    ----------
    path : str
        путь к файлу в формате txt, который содержит значения сгенерированного сверточного ядра, 
        добавляемого к исходным
    init_kernel : list
        значения исходного/ых ядер, к которым происходит добавление. 
        Это может быть либо переменная, которой присвоено значение исходных ядер, 
        либо сами значения, полученные после преобразования с помощью функции txt_kernel:
        [[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]], [[[7]], [[8]], [[9]]]].

    Пример использования:
    wdl.add_kernel('vertical.txt', w1)

    '''
    ff = open(path, 'r')
    
    w1 = init_kernel

    weights = []

    for i in ff:
        weights.append(i.strip().split(','))

    for j in weights:
        j.pop()
    
    for ii in weights:
        for i1, elem in enumerate(ii):
            ii[i1] = int(elem)
                               
    for i in range(len(w1)):
        for j in range(len(w1)):
            w1[i][j][0].append(weights[i][j])

    with open('kernel.txt', 'w') as file:
        file.write(str(w1))
        
    return w1

def horizontal(dim, values):
    '''
    Функция для генерации линейного сверточного ядра горизонтального типа

    Параметры
    ----------
    dim : int
        размерность генерируемой матрицы 
    values : list
        список значений, которые будут использоваться для заполнения матрицы.

    Пример использования:
    wdl.horizontal(32, [-7,-5,-3,-1,1,3,5,7])

    '''
    horizontal = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(values[i % len(values)])
        horizontal.append(row)
        
    with open('horizontal.txt', 'w') as f:
        for row in horizontal:
            f.write(','.join(str(x) for x in row) + ',\n')
    return horizontal

def vertical(dim, values):
    '''
    Функция для генерации линейного сверточного ядра вертикального типа

    Параметры
    ----------
    dim : int
        размерность генерируемой матрицы 
    values : list
        список значений, которые будут использоваться для заполнения матрицы.

    Пример использования:
    wdl.vertical(32, [-7,-5,-3,-1,1,3,5,7])

    '''
    vertical = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(values[j % len(values)])
        vertical.append(row)
        
    with open('vertical.txt', 'w') as f:    
        for row in vertical:
            f.write(','.join(str(x) for x in row) + ',\n')
    return vertical
	
def diagonal(up, down, d, dimension):
    '''
    Функция для генерации линейного сверточного ядра диагонального типа

    Parameters
    ----------
    up : list
        список значений, которые будут размещены над диагональю выходной матрицы
    down : list
        список значений, которые будут размещены ниже диагонали
    d : int
        значение, из которого состоит диагональ.
    dimension : int
        размерность генерируемого ядра

    Пример использования:
    wdl.diagonal([-7, -5, -3, -1], [1, 3, 5, 7], 1, 32)	

    '''
    matrix = [[0 for i in range(dimension)] for j in range(dimension)] 
    for i in range(dimension):
        for j in range(dimension):
            if i == j: 
                matrix[i][j] = d
            elif i < j: 
                matrix[i][j] = up[(j-i-1) % len(up)]
            else: 
                matrix[i][j] = down[(i-j-1) % len(down)]
    
    with open('diagonal1.txt', 'w') as file:
        for i in matrix:
            for j in i:
                file.write(str(j) + ',')
            file.write('\n')
            
    matrix2 = np.fliplr(matrix)
    
    with open('diagonal2.txt', 'w') as file:
        for i in matrix2:
            for j in i:
                file.write(str(j) + ',')
            file.write('\n')
    
    return matrix
    
def show_kernel(N, name):
    '''
    Функция вывода коэффициентов сверточного ядра на экран пользователя

    Параметры
    ----------
    N : int
        номер слоя, для которого необходимо вывести значения весовых коэффициентов
    name : object
        объект модели нейронной сети
        название переменной, которой присвоена структура модели

    Пример использования:
    wdl.show_kernel(0, model)

    '''
    ff = name.layers[N].get_weights()

    for nn in range(len(ff[0][0][0][0])):
      print('\n' + f'Номер сверточного ядра: {nn}')
      for kk in range(len(ff[0][0][0])):
        print(f'-----------ЦВЕТОВОЙ КАНАЛ №{kk}-----------')
        for i in range(len(ff[0])):
          for j in range(len(ff[0][i])):
            print(ff[0][i][j][kk][nn], end=',')
          print()
        print('\n')
      
def txt_kernel_rgb(k1, k2, k3): 
    '''
    Функция преобразования сгенерированного ядра для вставки в структуру TensorFlow 
    (для цветных, трехканальных изображений)
    
    Параметры
    ----------
    k1 : str
        путь к текстовому файлу, содержащему сверточное ядро первого цветового канала R, записанное в следующем виде 
        (с запятыми в конце каждой строки):
        1,2,3,
        4,5,6,
        7,8,9,
    k2 : str
        путь к текстовому файлу, содержащему сверточное ядро второго цветового канала G
    k3 : str
        путь к текстовому файлу, содержащему сверточное ядро третьего цветового канала B

    Пример использования:
    wdl.txt_kernel_rgb('k1.txt', 'k2.txt', 'k3.txt')

    '''
    f1 = open(k1, 'r')
    f2 = open(k2, 'r')
    f3 = open(k3, 'r')
	
    w1 = []

    for i in f1:
        w1.append(i.strip().split(','))
    for j in w1:
        j.pop()

    for ii in w1:
        for i1, elem in enumerate(ii):
            ii[i1] = [[int(elem)]]

    w2 = []

    for i in f2:
        w2.append(i.strip().split(','))

    for j in w2:
        j.pop()

    for ii in w2:
        for i1, elem in enumerate(ii):
            ii[i1] = [int(elem)]

    for i in range(len(w1[0])):
        for j in range(len(w1[0])):
            w1[i][j].append(w2[i][j])

    w3 = []

    for i in f3:
        w3.append(i.strip().split(','))

    for j in w3:
        j.pop()

    for ii in w3:
        for i1, elem in enumerate(ii):
            ii[i1] = [int(elem)]

    for i in range(len(w1[0])):
        for j in range(len(w1[0])):
            w1[i][j].append(w3[i][j])
			
    return w1

def add_kernel_rgb(k1, k2, k3, w):
    '''
    Функция позволяет устанавливать дополнительные ядра сверточного слоя к уже установленным исходным
    (для цветных, трехканальных изображений)

    Параметры
    ----------
    path : str
        путь к файлу в формате txt, который содержит значения сгенерированного сверточного ядра, 
        добавляемого к исходным
    init_kernel : list
        значения исходного/ых ядер, к которым происходит добавление. 
        Это может быть либо переменная, которой присвоено значение исходных ядер, 
        либо сами значения, полученные после преобразования с помощью функции txt_kernel_rgb

    Параметры
    ----------
    k1 : str
        путь к файлу в формате txt, который содержит значения сгенерированного сверточного ядра первого цветового канала R, 
        добавляемого к исходным в данном цветовом канале
    k2 : str
        путь к файлу в формате txt, который содержит значения сгенерированного сверточного ядра второго цветового канала G, 
        добавляемого к исходным в данном цветовом канале
    k3 : str
        путь к файлу в формате txt, который содержит значения сгенерированного сверточного ядра первого цветового канала B, 
        добавляемого к исходным в данном цветовом канале
    w : list
        значения исходного/ых ядер, к которым происходит добавление. 
        Это может быть либо переменная, которой присвоено значение исходных ядер, 
        либо сами значения, полученные после преобразования с помощью функции txt_kernel_rgb

    Пример использования:
    wdl.add_kernel_rgb('k1.txt', 'k2.txt', 'k3.txt', w1)

    '''
    f1 = open(k1, 'r')
    f2 = open(k2, 'r')
    f3 = open(k3, 'r')
	
    w11 = []

    for i in f1:
        w11.append(i.strip().split(','))

    for j in w11:
        j.pop()

    for ii in w11:
        for i1, elem in enumerate(ii):
            ii[i1] = int(elem)

    w22 = []

    for i in f2:
        w22.append(i.strip().split(','))

    for j in w22:
        j.pop()

    for ii in w22:
        for i1, elem in enumerate(ii):
            ii[i1] = int(elem)

    w33 = []

    for i in f3:
        w33.append(i.strip().split(','))

    for j in w33:
        j.pop()

    for ii in w33:
        for i1, elem in enumerate(ii):
            ii[i1] = int(elem)

    for i in range(len(w11)):
        for j in range(len(w11)):
            w[i][j][0].append(w11[i][j])
	########################################
    for i in range(len(w11)):
        for j in range(len(w11)):
            w[i][j][1].append(w22[i][j])
	########################################
    for i in range(len(w11)):
        for j in range(len(w11)):
            w[i][j][2].append(w33[i][j])
	########################################
	
    return w
      
def select_top(path, label):
    '''
    Функция позволяет получить грубое первоначальное приближение вектора весов

    Параметры
    ----------
    path : str
        путь к целочисленной обучающей выборке в формате csv
    label : str
        название столбца с метками классов

    Пример использования:
    wdl.select_top('KAHRAMAN_train.csv', 'UNS')

    '''
    all_classes = pd.read_csv(path)
    all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

    uniq = pd.unique(all_classes[[label]].values.ravel('K'))
    uniq = list(uniq)
    
    
    for o in range(len(uniq)):
        uniq1 = uniq.copy()
        z = o
        top = uniq1[z:z+1]
        uniq1.remove(top[0])
        other = uniq1.copy()

        class_top = all_classes[all_classes[label].isin(top)] 
        other_classes = all_classes[all_classes[label].isin(other)] 


        class_top['Scalar_calc'] = ""
        other_classes['Scalar_calc'] = ""

        class_top.to_csv('class_top.csv')
        other_classes.to_csv('other_classes.csv')

        class_top = pd.read_csv('class_top.csv')
        other_classes = pd.read_csv('other_classes.csv')

        class_top.drop(class_top.columns[0], axis=1, inplace=True)
        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

        cl = other_classes.columns

        columns = []
        for j in range(1, len(cl)):
            if(cl[j]=='N'):
                break
            else:
                columns.append(cl[j])

        d = []
        for c1 in range(len(columns)):
            z = f'd{c1}'
            d.append(z)
    
        class_top_exp = 1
        other_classes_exp = -(len(class_top)/len(other_classes))

        for c2 in range(len(d)):
            class_top[d[c2]] = ''
            other_classes[d[c2]] = ''

        for i1 in range(len(d)):
            class_top[d[i1]] = class_top[columns[i1]]*class_top_exp
    
        for i2 in range(len(d)):
            other_classes[d[i2]] = other_classes[columns[i2]]*other_classes_exp

        sm1 = class_top.sum()
        sm2 = other_classes.sum()

        weights1 = []
        for u in range(len(d)):
            weights1.append(sm1[d[u]])

        weights2 = []
        for uu in range(len(d)):
            weights2.append(sm2[d[uu]]) 
   
        weights = [x+y for x, y in zip(weights1, weights2)]
 
        for i in range(len(class_top)):
            xx = class_top.iloc[i]
            x = []
            for u in range(len(columns)):
                x.append(xx[columns[u]])
        
            f = np.array(weights)
            g = np.array(x)
            scalar = np.dot(f, g)
    
            class_top['Scalar_calc'][i]=scalar
    
        class_top = class_top.sort_values(by='Scalar_calc', ascending=False)

        for j in range(len(other_classes)):
            xx_other = other_classes.iloc[j]
            x_other = []
            for uu in range(len(columns)):
                x_other.append(xx_other[columns[uu]])
        
            f1 = np.array(weights)
            g1 = np.array(x_other)
            scalar1 = np.dot(f1, g1)
            other_classes['Scalar_calc'][j]=scalar1
    
        other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)

        up = 0
        for i in class_top['Scalar_calc']:
            if(i > other_classes['Scalar_calc'].max()):
                up+=1
            else:
                break
   
        other_classes.to_csv('other_classes.csv')
        other_classes = pd.read_csv('other_classes.csv')

        li = len(other_classes[label]) - 1
        
        lastindex_class_other_classes = other_classes[label][li]

        if((len(other_classes[label]))==1):
          down = 1
        else:
          down = 0
          for i in range(len(other_classes)-1, 0, -1):
            if(other_classes[label][i]==lastindex_class_other_classes):
                down+=1
            else:
                break

        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

        sum_up_down = up + down
    
        class_top_cut = class_top.iloc[up:]
        other_classes_cut = other_classes.iloc[:-down]
        
        del class_top_cut['Scalar_calc']
        del other_classes_cut['Scalar_calc']
        
        for i1 in range(len(d)):
            del class_top_cut[d[i1]] 
        
        for i2 in range(len(d)):
            del other_classes_cut[d[i2]]
            
        print('Количество отсеченных сверху = ', up)
        print("Количество НЕотсеченных сверху = ", len(class_top_cut))
        print('Количество отсеченных снизу = ', down)
        print("Количество НЕотсеченных снизу = ", len(other_classes_cut))
        print('===Нижняя категория: ', lastindex_class_other_classes)
        print('СУММА отсеченных сверху и снизу = ', sum_up_down)
        print('TOP - ', top)
        print('OTHERS - ', other)    
        print('+++++++++++++++++++++++++')
    
        os.mkdir(f'train_{top}{other}')

        class_top_cut.to_csv(f'train_{top}{other}/class_top_{top}{other}.csv')
        other_classes_cut.to_csv(f'train_{top}{other}/other_classes_{top}{other}.csv')
        
        with open(f'train_{top}{other}/w_{top}{other}.txt', 'w') as file:
            file.write(str(weights))
    
        oldpwd = os.getcwd()
        os.chdir(f"train_{top}{other}/")
    
        files = glob.glob("*.csv")
        combined = pd.DataFrame()
        for file in files:
            data = pd.read_csv(file)
            combined = pd.concat([combined, data])
        combined.drop(combined.columns[0], axis=1, inplace=True)
        combined.to_csv(f'train_{top}{other}.csv')
    
        os.chdir(oldpwd)

def grad_boost(path, t, oth, l, s, from_initial, weights_main):
    '''
    Вспомогательная функция для корректировки весовых коэффициентов с помощью градиентного уточнения, одиночный проход по вектору

    Параметры
    ----------
    path : str
        путь к обучающей выборке, подаваемой на вход функции select_top
    t : list
        метка выбранного на этапе первоначального приближения целевого класса
    oth : list
        список, содержащий метки оставшихся классов, которые не относятся к целевому
    l : str
        название столбца с метками классов
    s : int
        шаг изменения значений элементов вектора весов, шаг градиентного уточнения;
    from_initial : int
        1, если вектор берется из select_top (грубое приближение), 
        0 - если вектор снова корректируется (после grad_boost)
    weights_main : list
        вектор весов для корректировки

    '''
    weights_all = []
    
    for zz_i in range(len(weights_main)):
        
        
        zz = zz_i
    
        step = s

        label = l

        all_classes = pd.read_csv(path)#(path)#('wine_train.csv')
        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

        top = t
        other = oth

        class_top = all_classes[all_classes[label].isin(top)] 
        other_classes = all_classes[all_classes[label].isin(other)] 

        class_top['Scalar_calc'] = ""
        other_classes['Scalar_calc'] = ""

        class_top.to_csv('class_top.csv')
        other_classes.to_csv('other_classes.csv')

        class_top = pd.read_csv('class_top.csv')
        other_classes = pd.read_csv('other_classes.csv')

        class_top.drop(class_top.columns[0], axis=1, inplace=True)
        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

        cl = other_classes.columns

        columns = []
        for j in range(1, len(cl)):
            if(cl[j]=='N'):
                break
            else:
                columns.append(cl[j])

        w1 = weights_main

        weights = []
        if(from_initial == 1):
            for element in range(len(w1)):
                weights.append(w1[element]/(2*len(class_top)))
        else:
            for element in range(len(w1)):
                weights.append(w1[element])#/(2*len(class_top)))

        sign_step = [1, -1]

        step = step*sign_step[0]    

        for i in range(len(class_top)):
            xx = class_top.iloc[i]
            x = []
            for u in range(len(columns)):
                x.append(xx[columns[u]])
        
            f = np.array(weights)
            g = np.array(x)
            scalar = np.dot(f, g)
    
            class_top['Scalar_calc'][i]=scalar
    
        class_top = class_top.sort_values(by='Scalar_calc', ascending=False)

        for j in range(len(other_classes)):
            xx_other = other_classes.iloc[j]
            x_other = []
            for uu in range(len(columns)):
                x_other.append(xx_other[columns[uu]])
        
            f1 = np.array(weights)
            g1 = np.array(x_other)
            scalar1 = np.dot(f1, g1)
            other_classes['Scalar_calc'][j]=scalar1
    
        other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)

        up_start = 0
        for i in class_top['Scalar_calc']:
            if(i > other_classes['Scalar_calc'].max()):
                up_start+=1
            else:
                break
   
        other_classes.to_csv('other_classes.csv')
        other_classes = pd.read_csv('other_classes.csv')

        li = len(other_classes[label]) - 1
        
        lastindex_class_other_classes = other_classes[label][li]
    
        down = 0
        for i in range(len(other_classes)-1, 0, -1):
            if(other_classes[label][i]==lastindex_class_other_classes):
                down+=1
            else:
                break

        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)


        class_top.to_csv('class_top.csv')
        class_top = pd.read_csv('class_top.csv')
        class_top.drop(class_top.columns[0], axis=1, inplace=True)
        
        print()
        
        if(up_start >= len(class_top)):
            distance_start = other_classes['Scalar_calc'].max()#-class_top['Scalar_calc'][up_start]
        else:
            distance_start = other_classes['Scalar_calc'].max()-class_top['Scalar_calc'][up_start]
    
        d1 = distance_start
        u1 = up_start
    
        weights[zz]+=step

        for i in range(len(class_top)):
            xx = class_top.iloc[i]
            x = []
            for u in range(len(columns)):
                x.append(xx[columns[u]])
        
            f = np.array(weights)
            g = np.array(x)
            scalar = np.dot(f, g)
    
            class_top['Scalar_calc'][i]=scalar
    
        class_top = class_top.sort_values(by='Scalar_calc', ascending=False)

        for j in range(len(other_classes)):
            xx_other = other_classes.iloc[j]
            x_other = []
            for uu in range(len(columns)):
                x_other.append(xx_other[columns[uu]])
        
            f1 = np.array(weights)
            g1 = np.array(x_other)
            scalar1 = np.dot(f1, g1)
            other_classes['Scalar_calc'][j]=scalar1
    
        other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)

        up = 0
        for i in class_top['Scalar_calc']:
            if(i > other_classes['Scalar_calc'].max()):
                up+=1
            else:
                break
   
        other_classes.to_csv('other_classes.csv')
        other_classes = pd.read_csv('other_classes.csv')

        li = len(other_classes[label]) - 1
        
        lastindex_class_other_classes = other_classes[label][li]
    
        down = 0
        for i in range(len(other_classes)-1, 0, -1):
            if(other_classes[label][i]==lastindex_class_other_classes):
                down+=1
            else:
                break

        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

        sum_up_down = up + down

        class_top.to_csv('class_top.csv')
        class_top = pd.read_csv('class_top.csv')
        class_top.drop(class_top.columns[0], axis=1, inplace=True)
        
        #if(len(class_top) >= up):
        if(up >= len(class_top)):
            distance_current = other_classes['Scalar_calc'].max()#-class_top['Scalar_calc'][up]
        else:
            distance_current = other_classes['Scalar_calc'].max()-class_top['Scalar_calc'][up]

        if(distance_current > distance_start):
            step = step*sign_step[1]
        if(up < up_start):
            step = step*sign_step[1]
        else:
            step = step*sign_step[0]

        all_classes = pd.read_csv(path)#(path)#('wine_train.csv')
        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

        class_top = all_classes[all_classes[label].isin(top)] 
        other_classes = all_classes[all_classes[label].isin(other)] 

        class_top['Scalar_calc'] = ""
        other_classes['Scalar_calc'] = ""

        class_top.to_csv('class_top.csv')
        other_classes.to_csv('other_classes.csv')

        class_top = pd.read_csv('class_top.csv')
        other_classes = pd.read_csv('other_classes.csv')

        class_top.drop(class_top.columns[0], axis=1, inplace=True)
        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

        w2 = w1
    
        weights = []
        if(from_initial == 1):
            for element in range(len(w2)):
                weights.append(w2[element]/(2*len(class_top)))
        else:
            for element in range(len(w2)):
                weights.append(w2[element])#/(2*len(class_top)))
    
        distance_start = d1

        up_start = u1

        while True:
            dd = distance_current

            for i in range(len(class_top)):
                xx = class_top.iloc[i]
                x = []
                for u in range(len(columns)):
                    x.append(xx[columns[u]])
        
                f = np.array(weights)
                g = np.array(x)
                scalar = np.dot(f, g)
    
                class_top['Scalar_calc'][i]=scalar
    
            class_top = class_top.sort_values(by='Scalar_calc', ascending=False)

            for j in range(len(other_classes)):
                xx_other = other_classes.iloc[j]
                x_other = []
                for uu in range(len(columns)):
                    x_other.append(xx_other[columns[uu]])
        
                f1 = np.array(weights)
                g1 = np.array(x_other)
                scalar1 = np.dot(f1, g1)
                other_classes['Scalar_calc'][j]=scalar1
    
            other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)
        
            up = 0
            for i in class_top['Scalar_calc']:
                if(i > other_classes['Scalar_calc'].max()):
                    up+=1
                else:
                    break
   
            other_classes.to_csv('other_classes.csv')
            other_classes = pd.read_csv('other_classes.csv')

            li = len(other_classes[label]) - 1
        
            lastindex_class_other_classes = other_classes[label][li]

            down = 0
            for i in range(len(other_classes)-1, 0, -1):
                if(other_classes[label][i]==lastindex_class_other_classes):
                    down+=1
                else:
                    break

            other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

            sum_up_down = up + down

            class_top.to_csv('class_top.csv')
            class_top = pd.read_csv('class_top.csv')
            class_top.drop(class_top.columns[0], axis=1, inplace=True)
            
            if(up >= len(class_top)):
                distance_current = other_classes['Scalar_calc'].max()#-class_top['Scalar_calc'][up]
            else:
                distance_current = other_classes['Scalar_calc'].max()-class_top['Scalar_calc'][up]
        
            print('Количество отсеченных сверху = ', up)
            print('Количество отсеченных снизу = ', down)  
            print('DISTANCE = ', distance_current)
            print('\nВЕС: ', weights)
            print('+++++++++++++++++++++++++')
            print()
            print()

            if(up > up_start) and (distance_current > distance_start):
                distance_start = distance_current
                up_start = up
            if(up > up_start) and (distance_current < distance_start):
                distance_start = distance_current
                up_start = up
            if(up == up_start) and (distance_current < distance_start):
                distance_start = distance_current
                up_start = up
            if(((up <= up_start) and (distance_current > distance_start) or (dd == distance_start)) or ((up < up_start) and (distance_current < distance_start) or (dd == distance_start))):
                weights[zz] = weights[zz] - step
            
                ####
            
                for i in range(len(class_top)):
                    xx = class_top.iloc[i]
                    x = []
                    for u in range(len(columns)):
                        x.append(xx[columns[u]])
                    f = np.array(weights)
                    g = np.array(x)
                    scalar = np.dot(f, g)
                    class_top['Scalar_calc'][i]=scalar
                class_top = class_top.sort_values(by='Scalar_calc', ascending=False)
            
                for j in range(len(other_classes)):
                    xx_other = other_classes.iloc[j]
                    x_other = []
                    for uu in range(len(columns)):
                        x_other.append(xx_other[columns[uu]])
                    f1 = np.array(weights)
                    g1 = np.array(x_other)
                    scalar1 = np.dot(f1, g1)
                    other_classes['Scalar_calc'][j]=scalar1
                other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)
            
                up_previous = 0
                for i in class_top['Scalar_calc']:
                    if(i > other_classes['Scalar_calc'].max()):
                        up_previous+=1
                    else:
                        break
                other_classes.to_csv('other_classes.csv')
                other_classes = pd.read_csv('other_classes.csv')
            
                li = len(other_classes[label]) - 1
            
                lastindex_class_other_classes = other_classes[label][li]

                if((len(other_classes[label]))==1):
                  down_previous = 1
                else:
                  down_previous = 0
                  for i in range(len(other_classes)-1, 0, -1):
                    if(other_classes[label][i]==lastindex_class_other_classes):
                        down_previous+=1
                    else:
                        break
                other_classes.drop(other_classes.columns[0], axis=1, inplace=True)
                
                print(up_previous)
                print(down_previous)
                
                distance_previous = distance_start
                
                xxx = [weights, up_previous, down_previous, distance_previous]
                weights_all.append(xxx)

                break

            weights[zz]+=step  
        
            print()
            
            
    print(weights_all)
    return weights_all

def grad_boost_save_files(path, t, oth, l, s, from_initial, weights_main):
    '''
    Вспомогательная функция для сохранения усеченных обучающих выборок и скорректированных векторов весов после градиентного уточнения

    Параметры
    ----------
    path : str
        путь к обучающей выборке, подаваемой на вход функции select_top
    t : list
        метка выбранного на этапе первоначального приближения целевого класса
    oth : list
        список, содержащий метки оставшихся классов, которые не относятся к целевому
    l : str
        название столбца с метками классов
    s : int
        шаг изменения значений элементов вектора весов, шаг градиентного уточнения;
    from_initial : int
        1, если вектор берется из select_top (грубое приближение), 
        0 - если вектор снова корректируется (после grad_boost)
    weights_main : list
        вектор весов для корректировки

    '''
    weights_all = []
    
    for zz_i in range(len(weights_main)):
        
        
        zz = zz_i
    
        step = s

        label = l

        all_classes = pd.read_csv(path)#(path)#('wine_train.csv')
        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

        top = t
        other = oth

        class_top = all_classes[all_classes[label].isin(top)] 
        other_classes = all_classes[all_classes[label].isin(other)] 

        class_top['Scalar_calc'] = ""
        other_classes['Scalar_calc'] = ""

        class_top.to_csv('class_top.csv')
        other_classes.to_csv('other_classes.csv')

        class_top = pd.read_csv('class_top.csv')
        other_classes = pd.read_csv('other_classes.csv')

        class_top.drop(class_top.columns[0], axis=1, inplace=True)
        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

        cl = other_classes.columns

        columns = []
        for j in range(1, len(cl)):
            if(cl[j]=='N'):
                break
            else:
                columns.append(cl[j])

        w1 = weights_main

        weights = []
        if(from_initial == 1):
            for element in range(len(w1)):
                weights.append(w1[element]/(2*len(class_top)))
        else:
            for element in range(len(w1)):
                weights.append(w1[element])#/(2*len(class_top)))

        sign_step = [1, -1]

        step = step*sign_step[0]    

        for i in range(len(class_top)):
            xx = class_top.iloc[i]
            x = []
            for u in range(len(columns)):
                x.append(xx[columns[u]])
        
            f = np.array(weights)
            g = np.array(x)
            scalar = np.dot(f, g)
    
            class_top['Scalar_calc'][i]=scalar
    
        class_top = class_top.sort_values(by='Scalar_calc', ascending=False)

        for j in range(len(other_classes)):
            xx_other = other_classes.iloc[j]
            x_other = []
            for uu in range(len(columns)):
                x_other.append(xx_other[columns[uu]])
        
            f1 = np.array(weights)
            g1 = np.array(x_other)
            scalar1 = np.dot(f1, g1)
            other_classes['Scalar_calc'][j]=scalar1
    
        other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)

        up_start = 0
        for i in class_top['Scalar_calc']:
            if(i > other_classes['Scalar_calc'].max()):
                up_start+=1
            else:
                break
   
        other_classes.to_csv('other_classes.csv')
        other_classes = pd.read_csv('other_classes.csv')

        li = len(other_classes[label]) - 1
        
        lastindex_class_other_classes = other_classes[label][li]
    
        down = 0
        for i in range(len(other_classes)-1, 0, -1):
            if(other_classes[label][i]==lastindex_class_other_classes):
                down+=1
            else:
                break

        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)


        class_top.to_csv('class_top.csv')
        class_top = pd.read_csv('class_top.csv')
        class_top.drop(class_top.columns[0], axis=1, inplace=True)
        
        print()
        
        if(up_start >= len(class_top)):
            distance_start = other_classes['Scalar_calc'].max()#-class_top['Scalar_calc'][up_start]
        else:
            distance_start = other_classes['Scalar_calc'].max()-class_top['Scalar_calc'][up_start]
    
        d1 = distance_start
        u1 = up_start
    
        weights[zz]+=step

        for i in range(len(class_top)):
            xx = class_top.iloc[i]
            x = []
            for u in range(len(columns)):
                x.append(xx[columns[u]])
        
            f = np.array(weights)
            g = np.array(x)
            scalar = np.dot(f, g)
    
            class_top['Scalar_calc'][i]=scalar
    
        class_top = class_top.sort_values(by='Scalar_calc', ascending=False)

        for j in range(len(other_classes)):
            xx_other = other_classes.iloc[j]
            x_other = []
            for uu in range(len(columns)):
                x_other.append(xx_other[columns[uu]])
        
            f1 = np.array(weights)
            g1 = np.array(x_other)
            scalar1 = np.dot(f1, g1)
            other_classes['Scalar_calc'][j]=scalar1
    
        other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)

        up = 0
        for i in class_top['Scalar_calc']:
            if(i > other_classes['Scalar_calc'].max()):
                up+=1
            else:
                break
   
        other_classes.to_csv('other_classes.csv')
        other_classes = pd.read_csv('other_classes.csv')

        li = len(other_classes[label]) - 1
        
        lastindex_class_other_classes = other_classes[label][li]
    
        down = 0
        for i in range(len(other_classes)-1, 0, -1):
            if(other_classes[label][i]==lastindex_class_other_classes):
                down+=1
            else:
                break

        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

        sum_up_down = up + down

        class_top.to_csv('class_top.csv')
        class_top = pd.read_csv('class_top.csv')
        class_top.drop(class_top.columns[0], axis=1, inplace=True)
        
        #if(len(class_top) >= up):
        if(up >= len(class_top)):
            distance_current = other_classes['Scalar_calc'].max()#-class_top['Scalar_calc'][up]
        else:
            distance_current = other_classes['Scalar_calc'].max()-class_top['Scalar_calc'][up]

        if(distance_current > distance_start):
            step = step*sign_step[1]
        if(up < up_start):
            step = step*sign_step[1]
        else:
            step = step*sign_step[0]

        all_classes = pd.read_csv(path)#(path)#('wine_train.csv')
        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

        class_top = all_classes[all_classes[label].isin(top)] 
        other_classes = all_classes[all_classes[label].isin(other)] 

        class_top['Scalar_calc'] = ""
        other_classes['Scalar_calc'] = ""

        class_top.to_csv('class_top.csv')
        other_classes.to_csv('other_classes.csv')

        class_top = pd.read_csv('class_top.csv')
        other_classes = pd.read_csv('other_classes.csv')

        class_top.drop(class_top.columns[0], axis=1, inplace=True)
        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

        w2 = w1
    
        weights = []
        if(from_initial == 1):
            for element in range(len(w2)):
                weights.append(w2[element]/(2*len(class_top)))
        else:
            for element in range(len(w2)):
                weights.append(w2[element])#/(2*len(class_top)))
    
        distance_start = d1

        up_start = u1

        while True:
            dd = distance_current

            for i in range(len(class_top)):
                xx = class_top.iloc[i]
                x = []
                for u in range(len(columns)):
                    x.append(xx[columns[u]])
        
                f = np.array(weights)
                g = np.array(x)
                scalar = np.dot(f, g)
    
                class_top['Scalar_calc'][i]=scalar
    
            class_top = class_top.sort_values(by='Scalar_calc', ascending=False)

            for j in range(len(other_classes)):
                xx_other = other_classes.iloc[j]
                x_other = []
                for uu in range(len(columns)):
                    x_other.append(xx_other[columns[uu]])
        
                f1 = np.array(weights)
                g1 = np.array(x_other)
                scalar1 = np.dot(f1, g1)
                other_classes['Scalar_calc'][j]=scalar1
    
            other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)
        
            #up_previous = up
            up = 0
            for i in class_top['Scalar_calc']:
                if(i > other_classes['Scalar_calc'].max()):
                    up+=1
                else:
                    break
   
            other_classes.to_csv('other_classes.csv')
            other_classes = pd.read_csv('other_classes.csv')

            li = len(other_classes[label]) - 1
        
            lastindex_class_other_classes = other_classes[label][li]

            down = 0
            for i in range(len(other_classes)-1, 0, -1):
                if(other_classes[label][i]==lastindex_class_other_classes):
                    down+=1
                else:
                    break

            other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

            sum_up_down = up + down

            class_top.to_csv('class_top.csv')
            class_top = pd.read_csv('class_top.csv')
            class_top.drop(class_top.columns[0], axis=1, inplace=True)
            
            #if(len(class_top) >= up):
            if(up >= len(class_top)):
                distance_current = other_classes['Scalar_calc'].max()#-class_top['Scalar_calc'][up]
            else:
                distance_current = other_classes['Scalar_calc'].max()-class_top['Scalar_calc'][up]
        
            print('Количество отсеченных сверху = ', up)
            print('Количество отсеченных снизу = ', down)  
            print('DISTANCE = ', distance_current)
            print('\nВЕС: ', weights)
            print('+++++++++++++++++++++++++')
            print()
            print()

            if(up > up_start) and (distance_current > distance_start):
                distance_start = distance_current
                up_start = up
            if(up > up_start) and (distance_current < distance_start):
                distance_start = distance_current
                up_start = up
            if(up == up_start) and (distance_current < distance_start):
                distance_start = distance_current
                up_start = up
            if(((up <= up_start) and (distance_current > distance_start) or (dd == distance_start)) or ((up < up_start) and (distance_current < distance_start) or (dd == distance_start))):
                weights[zz] = weights[zz] - step
            
                ####
            
                for i in range(len(class_top)):
                    xx = class_top.iloc[i]
                    x = []
                    for u in range(len(columns)):
                        x.append(xx[columns[u]])
                    f = np.array(weights)
                    g = np.array(x)
                    scalar = np.dot(f, g)
                    class_top['Scalar_calc'][i]=scalar
                class_top = class_top.sort_values(by='Scalar_calc', ascending=False)
            
                for j in range(len(other_classes)):
                    xx_other = other_classes.iloc[j]
                    x_other = []
                    for uu in range(len(columns)):
                        x_other.append(xx_other[columns[uu]])
                    f1 = np.array(weights)
                    g1 = np.array(x_other)
                    scalar1 = np.dot(f1, g1)
                    other_classes['Scalar_calc'][j]=scalar1
                other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)
            
                up_previous = 0
                for i in class_top['Scalar_calc']:
                    if(i > other_classes['Scalar_calc'].max()):
                        up_previous+=1
                    else:
                        break
                other_classes.to_csv('other_classes.csv')
                other_classes = pd.read_csv('other_classes.csv')
            
                li = len(other_classes[label]) - 1
            
                lastindex_class_other_classes = other_classes[label][li]

                if((len(other_classes[label]))==1):
                  down_previous = 1
                else:
                  down_previous = 0
                  for i in range(len(other_classes)-1, 0, -1):
                    if(other_classes[label][i]==lastindex_class_other_classes):
                        down_previous+=1
                    else:
                        break
                other_classes.drop(other_classes.columns[0], axis=1, inplace=True)
                
                print(up_previous)
                print(down_previous)
                
                distance_previous = distance_start
            
                ####
                
                class_top_cut = class_top.iloc[up_previous:]
                other_classes_cut = other_classes.iloc[:-down_previous]
            
                del class_top_cut['Scalar_calc']
                del other_classes_cut['Scalar_calc']
            
                os.mkdir(f'train_{zz}')
                class_top_cut.to_csv(f'train_{zz}/class_top_{zz}.csv')
                other_classes_cut.to_csv(f'train_{zz}/other_classes_{zz}.csv')
            
                with open(f'train_{zz}/w_{zz}.txt', 'w') as file:
                    file.write(str(weights) + '\n' + str(up_previous) + '\n' + str(down_previous) + '\n' + str(distance_previous))
            
                oldpwd = os.getcwd()
                os.chdir(f"train_{zz}/")
            
                files = glob.glob("*.csv")
                combined = pd.DataFrame()
                for file in files:
                    data = pd.read_csv(file)
                    combined = pd.concat([combined, data])
                combined.drop(combined.columns[0], axis=1, inplace=True)
                combined.to_csv(f'train_{zz}.csv')
            
                os.chdir(oldpwd)
                
                xxx = [weights, up_previous, down_previous, distance_previous]
                weights_all.append(xxx)

                break

            weights[zz]+=step  
        
            print()
            
    print(weights_all)
    return weights_all

def select_top_binary(path, label):
    '''
    Функция позволяет получить грубое первоначальное приближение вектора весов для бинарной обучающей выборки

    Параметры
    ----------
    path : str
        путь к целочисленной бинарной обучающей выборке в формате csv
    label : str
        название столбца с метками классов

    Пример использования:
    wdl.select_top_binary('train_6.csv', 'UNS')

    '''
    all_classes_main = pd.read_csv(path)
    all_classes_main.drop(all_classes_main.columns[0], axis=1, inplace=True)
    
    uniq = pd.unique(all_classes_main[[label]].values.ravel('K'))
    uniq = list(uniq)
    
    for o in range(len(uniq)):
        uniq1 = uniq.copy()
        z = o
        top = uniq1[z:z+1]
        uniq1.remove(top[0])
        other = uniq1.copy()
    
        class_top = all_classes_main[all_classes_main[label].isin(top)] 
        other_classes = all_classes_main[all_classes_main[label].isin(other)] 
    
        class_top['Scalar_calc'] = ""
        other_classes['Scalar_calc'] = ""

        class_top.to_csv('class_top.csv')
        other_classes.to_csv('other_classes.csv')

        class_top = pd.read_csv('class_top.csv')
        other_classes = pd.read_csv('other_classes.csv')

        class_top.drop(class_top.columns[0], axis=1, inplace=True)
        other_classes.drop(other_classes.columns[0], axis=1, inplace=True)

        cl = all_classes_main.columns

        columns = []
        for j in range(1, len(cl)):
            if(cl[j]=='N'):
                break
            else:
                columns.append(cl[j])

        d = []
        for c1 in range(len(columns)):
            z = f'd{c1}'
            d.append(z)

        class_top_exp = 1
        other_classes_exp = 1-(len(class_top)/len(other_classes))

        for c2 in range(len(d)):
            class_top[d[c2]] = ''
            other_classes[d[c2]] = ''
        
        for i1 in range(len(d)):
            class_top[d[i1]] = class_top[columns[i1]]*class_top_exp
    
        for i2 in range(len(d)):
            other_classes[d[i2]] = other_classes[columns[i2]]*other_classes_exp
    
        sm1 = class_top.sum()
        sm2 = other_classes.sum()
    
        weights1 = []
        for u in range(len(d)):
            weights1.append(sm1[d[u]])

        weights2 = []
        for uu in range(len(d)):
            weights2.append(sm2[d[uu]]) 
        
        weights = [x+y for x, y in zip(weights1, weights2)]
    
        for i in range(len(class_top)):
            xx = class_top.iloc[i]
            x = []
            for u in range(len(columns)):
                x.append(xx[columns[u]])
        
            f = np.array(weights)
            g = np.array(x)
            scalar = np.dot(f, g)
    
            class_top['Scalar_calc'][i]=scalar

        for j in range(len(other_classes)):
            xx_other = other_classes.iloc[j]
            x_other = []
            for uu in range(len(columns)):
                x_other.append(xx_other[columns[uu]])
        
            f1 = np.array(weights)
            g1 = np.array(x_other)
            scalar1 = np.dot(f1, g1)
            other_classes['Scalar_calc'][j]=scalar1
    
        all_classes = pd.concat([class_top, other_classes])
    
        all_classes = all_classes.sort_values(by='Scalar_calc', ascending=False)
        all_classes.to_csv('all_classes.csv')
        all_classes = pd.read_csv('all_classes.csv')
    
        firstindex = all_classes[label][0]

        up = 0
        for i in range(len(all_classes)):
            if(all_classes[label][i]==firstindex):
                up+=1
            else:
                break

        li = len(all_classes[label])-1
        
        lastindex_class_binary_classes = all_classes[label][li]

        if((len(all_classes[label]))==1):
            down = 1
        else:
            down = 0
            for i in range(len(all_classes)-1, 0, -1):
                if(all_classes[label][i]==lastindex_class_binary_classes):
                    down+=1
                else:
                    break

        sum_up_down = up + down
    
        dd1 = all_classes['Scalar_calc'][up-1]
        dd2 = all_classes['Scalar_calc'][up]
        distance = dd1 - dd2

        all_classes_upcut = all_classes.iloc[up:]
        all_classes_allcut = all_classes_upcut.iloc[:-down]
        
        del all_classes_allcut['Scalar_calc']
        
        for i1 in range(len(d)):
            del all_classes_allcut[d[i1]] 
    
        all_classes_allcut.drop(all_classes_allcut.columns[0], axis=1, inplace=True)
        all_classes_allcut.to_csv(f'train_{top}{other}.csv')
        
        with open(f'w_{top}{other}.txt', 'w') as file:
            file.write(str(weights))
            
        print('Количество отсеченных сверху = ', up)
        print('Количество отсеченных снизу = ', down)
        print('===Верхняя категория: ', firstindex)
        print('===Нижняя категория: ', lastindex_class_binary_classes)
        print('TOP - ', top)
        print('OTHER - ', other) 
        print('СУММА отсеченных сверху и снизу = ', sum_up_down)
        print(weights)
        print('DISTANCE = ', distance)
        print('+++++++++++++++++++++++++++++++++++++++')
        print()      

def grad_binary(path, t, oth, l, s, weights_main):
    '''
    Функция осуществляет градиентное уточнение полученного после первоначального приближения вектора весов 
    для бинарной обучающей выборки

    Параметры
    ----------
    path : str
        путь к бинарной обучающей выборке, подаваемой на вход функции select_top
    t : list
        метка выбранного на этапе первоначального приближения целевого класса
    oth : list
        список, содержащий метки оставшихся классов, которые не относятся к целевому
    l : str
        название столбца с метками классов
    s : int
        шаг изменения значений элементов вектора весов, шаг градиентного уточнения
    weights_main : list
        вектор весов, значения которого необходимо изменить с целью получения большего количества 
        отсеченных экземпляров обучающей выборки

    Пример использования:
    wdl.grad_binary('train_6.csv', ['Low'], ['Middle'], 'UNS', 1, [1009.0, 48.0, -929.0, -1056.0, -13.0])

    '''
    from_initial = 1

    for zz_i in range(len(weights_main)):
        zz = zz_i
        step = s
        label = l
        top = t
        other = oth
        
        all_classes = pd.read_csv(path)
        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

        all_classes['Scalar_calc'] = ""

        all_classes.to_csv('all_classes.csv')

        all_classes = pd.read_csv('all_classes.csv')

        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

        cl = all_classes.columns

        columns = []
        for j in range(1, len(cl)):
            if(cl[j]=='N'):
                break
            else:
                columns.append(cl[j])
                
        w1 = weights_main

        weights = []
        if(from_initial == 1):
            for element in range(len(w1)):
                weights.append(w1[element])#/(2*len(class_top)))
        else:
            for element in range(len(w1)):
                weights.append(w1[element])
                
        sign_step = [1, -1]
        
        step = step*sign_step[0]  
            
        for i in range(len(all_classes)):
            xx = all_classes.iloc[i]
            x = []
            for u in range(len(columns)):
                x.append(xx[columns[u]])
        
            f = np.array(weights)
            g = np.array(x)
            scalar = np.dot(f, g)
    
            all_classes['Scalar_calc'][i]=scalar
    
        all_classes = all_classes.sort_values(by='Scalar_calc', ascending=False)

        all_classes.to_csv('all_classes.csv')
        all_classes = pd.read_csv('all_classes.csv')
        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

        firstindex = all_classes[label][0]
        
        if((len(all_classes[label]))==2):
            up_start = 1
        else:
            up_start = 0
            for i in range(len(all_classes)):
                if(all_classes[label][i]==firstindex):
                    up_start+=1
                else:
                    break

        li = len(all_classes[label])-1
        
        lastindex_class_binary_classes = all_classes[label][li]

        if((len(all_classes[label]))==2):
            down = 1
        else:
            down = 0
            for i in range(len(all_classes)-1, 0, -1):
                if(all_classes[label][i]==lastindex_class_binary_classes):
                    down+=1
                else:
                    break
                    
        all_classes_down = all_classes.sort_values(by='Scalar_calc', ascending=True)
        all_classes_down.to_csv('all_classes_down.csv')
        all_classes_down = pd.read_csv('all_classes_down.csv')
        all_classes_down.drop(all_classes_down.columns[0], axis=1, inplace=True)
        
        if(len(all_classes)!=2):
            dd1 = all_classes['Scalar_calc'][up_start-1]
            for i in range(down, 0, -1):
                dd2 = all_classes_down['Scalar_calc'][i+1]
            distance_start = dd1 - dd2
        else:
            dd1 = 0
            dd2 = 0
            distance_start = dd1 - dd2
        
        d1 = distance_start
        u1 = up_start
        
        weights[zz]+=step
        
        for i in range(len(all_classes)):
            xx = all_classes.iloc[i]
            x = []
            for u in range(len(columns)):
                x.append(xx[columns[u]])
        
            f = np.array(weights)
            g = np.array(x)
            scalar = np.dot(f, g)
    
            all_classes['Scalar_calc'][i]=scalar
    
        all_classes = all_classes.sort_values(by='Scalar_calc', ascending=False)
        
        all_classes.to_csv('all_classes.csv')
        all_classes = pd.read_csv('all_classes.csv')
        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

        firstindex = all_classes[label][0]
        
        if((len(all_classes[label]))==2):
            up = 1
        else:
            up = 0
            for i in range(len(all_classes)):
                if(all_classes[label][i]==firstindex):
                    up+=1
                else:
                    break

        li = len(all_classes[label])-1
        
        lastindex_class_binary_classes = all_classes[label][li]

        if((len(all_classes[label]))==2):
            down = 1
        else:
            down = 0
            for i in range(len(all_classes)-1, 0, -1):
                if(all_classes[label][i]==lastindex_class_binary_classes):
                    down+=1
                else:
                    break
        
        sum_up_down = up + down
        
        all_classes_down1 = all_classes.sort_values(by='Scalar_calc', ascending=True)
        all_classes_down1.to_csv('all_classes_down1.csv')
        all_classes_down1 = pd.read_csv('all_classes_down1.csv')
        all_classes_down1.drop(all_classes_down1.columns[0], axis=1, inplace=True)
        
        if(len(all_classes)!=2):
            dd1 = all_classes['Scalar_calc'][up-1]
            for i in range(down, 0, -1):
                dd2 = all_classes_down1['Scalar_calc'][i+1]
                #dd2 = all_classes['Scalar_calc'][up]
            distance_current = dd1 - dd2
        else:
            dd1 = 0
            dd2 = 0
            distance_current = dd1 - dd2
        
        if(distance_current > distance_start):
            step = step*sign_step[1]
        if(up < up_start):
            step = step*sign_step[1]
        else:
            step = step*sign_step[0]
            
        all_classes = pd.read_csv(path)
        
        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

        all_classes['Scalar_calc'] = ""

        all_classes.to_csv('all_classes.csv')

        all_classes = pd.read_csv('all_classes.csv')

        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)    
            
        w2 = w1
        
        weights = []
        if(from_initial == 1):
            for element in range(len(w2)):
                weights.append(w2[element])#/(2*len(class_top)))
        else:
            for element in range(len(w2)):
                weights.append(w2[element])
                
        distance_start = d1
        
        up_start = u1
        
        while True:
            
            dd = distance_current
            
            for i in range(len(all_classes)):
                xx = all_classes.iloc[i]
                x = []
                for u in range(len(columns)):
                    x.append(xx[columns[u]])
        
                f = np.array(weights)
                g = np.array(x)
                scalar = np.dot(f, g)
    
                all_classes['Scalar_calc'][i]=scalar
    
            all_classes = all_classes.sort_values(by='Scalar_calc', ascending=False)
        
            all_classes.to_csv('all_classes.csv')
            all_classes = pd.read_csv('all_classes.csv')
            all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

            firstindex = all_classes[label][0]
        
            if((len(all_classes[label]))==2):
                up = 1
            else:
                up = 0
                for i in range(len(all_classes)):
                    if(all_classes[label][i]==firstindex):
                        up+=1
                    else:
                        break

            li = len(all_classes[label])-1
        
            lastindex_class_binary_classes = all_classes[label][li]

            if((len(all_classes[label]))==2):
                down = 1
            else:
                down = 0
                for i in range(len(all_classes)-1, 0, -1):
                    if(all_classes[label][i]==lastindex_class_binary_classes):
                        down+=1
                    else:
                        break
        
            sum_up_down = up + down
            
            all_classes_down2 = all_classes.sort_values(by='Scalar_calc', ascending=True)
            all_classes_down2.to_csv('all_classes_down2.csv')
            all_classes_down2 = pd.read_csv('all_classes_down2.csv')
            all_classes_down2.drop(all_classes_down2.columns[0], axis=1, inplace=True)
            
            if(len(all_classes)!=2):
                dd1 = all_classes['Scalar_calc'][up-1]
                for i in range(down, 0, -1):
                    dd2 = all_classes_down2['Scalar_calc'][i+1]
                    #dd2 = all_classes['Scalar_calc'][up]
                distance_current = dd1 - dd2
            else:
                dd1 = 0
                dd2 = 0
                distance_current = dd1 - dd2
            
            print('Количество отсеченных сверху = ', up)
            print('Количество отсеченных снизу = ', down)  
            print('DISTANCE = ', distance_current)
            print('\nВЕС: ', weights)
            print('+++++++++++++++++++++++++')
            print()
            print()
            
            
            if(up > up_start) and (distance_current > distance_start):
                distance_start = distance_current
                up_start = up
            if(up > up_start) and (distance_current < distance_start):
                distance_start = distance_current
                up_start = up
            if(up == up_start) and (distance_current < distance_start):
                distance_start = distance_current
                up_start = up
            if(((up <= up_start) and (distance_current > distance_start) or (dd == distance_start)) or ((up < up_start) and (distance_current < distance_start) or (dd == distance_start))):
                weights[zz] = weights[zz] - step
                
                ###
            
                for i in range(len(all_classes)):
                    xx = all_classes.iloc[i]
                    x = []
                    for u in range(len(columns)):
                        x.append(xx[columns[u]])
        
                    f = np.array(weights)
                    g = np.array(x)
                    scalar = np.dot(f, g)
    
                    all_classes['Scalar_calc'][i]=scalar
    
                all_classes = all_classes.sort_values(by='Scalar_calc', ascending=False)
        
                all_classes.to_csv('all_classes.csv')
                all_classes = pd.read_csv('all_classes.csv')
                all_classes.drop(all_classes.columns[0], axis=1, inplace=True)

                firstindex = all_classes[label][0]
        
                if((len(all_classes[label]))==2):
                    up_previous = 1
                else:
                    up_previous = 0
                    for i in range(len(all_classes)):
                        if(all_classes[label][i]==firstindex):
                            up_previous+=1
                        else:
                            break

                li = len(all_classes[label])-1
        
                lastindex_class_binary_classes = all_classes[label][li]

                if((len(all_classes[label]))==2):
                    down_previous = 1
                else:
                    down_previous = 0
                    for i in range(len(all_classes)-1, 0, -1):
                        if(all_classes[label][i]==lastindex_class_binary_classes):
                            down_previous+=1
                        else:
                            break
        
                #sum_up_down = up + down
            
                all_classes_down3 = all_classes.sort_values(by='Scalar_calc', ascending=True)
                all_classes_down3.to_csv('all_classes_down3.csv')
                all_classes_down3 = pd.read_csv('all_classes_down3.csv')
                all_classes_down3.drop(all_classes_down3.columns[0], axis=1, inplace=True)
            
                if(len(all_classes)!=2):
                    dd1 = all_classes['Scalar_calc'][up-1]
                    for i in range(down_previous, 0, -1):
                        dd2 = all_classes_down3['Scalar_calc'][i+1]
                    #dd2 = all_classes['Scalar_calc'][up]
                    distance_previous = dd1 - dd2
                else:
                    dd1 = 0
                    dd2 = 0
                    distance_previous = dd1 - dd2
            
                all_classes = pd.read_csv(path)
                all_classes.drop(all_classes.columns[0], axis=1, inplace=True)
                all_classes['Scalar_calc'] = ""
                all_classes.to_csv('all_classes.csv')
                all_classes = pd.read_csv('all_classes.csv')
                all_classes.drop(all_classes.columns[0], axis=1, inplace=True)
            
                for i in range(len(all_classes)):
                    xx = all_classes.iloc[i]
                    x = []
                    for u in range(len(columns)):
                        x.append(xx[columns[u]])
                    f = np.array(weights)
                    g = np.array(x)
                    scalar = np.dot(f, g)
                    all_classes['Scalar_calc'][i]=scalar
                all_classes = all_classes.sort_values(by='Scalar_calc', ascending=False)
                all_classes.to_csv('all_classes.csv')
                all_classes = pd.read_csv('all_classes.csv')
                #all_classes.drop(all_classes.columns[0], axis=1, inplace=True)
                firstindex = all_classes[label][0]
                if((len(all_classes[label]))==2):
                    UP = 1
                else:
                    UP = 0
                    for i in range(len(all_classes)):
                        if(all_classes[label][i]==firstindex):
                            UP += 1
                        else:
                            break
                li = len(all_classes[label])-1
                lastindex_class_binary_classes = all_classes[label][li]
                if((len(all_classes[label]))==2):
                    DOWN = 1
                else:
                    DOWN = 0
                    for i in range(len(all_classes)-1, 0, -1):
                        if(all_classes[label][i]==lastindex_class_binary_classes):
                            DOWN += 1
                        else:
                            break
            
                all_classes_upcut = all_classes.iloc[UP:]
                all_classes_allcut = all_classes_upcut.iloc[:-DOWN]
        
                del all_classes_allcut['Scalar_calc']
            
                os.mkdir(f'train_{zz}')
    
                all_classes_allcut.drop(all_classes_allcut.columns[0], axis=1, inplace=True)
                all_classes_allcut.to_csv(f'train_{zz}/all_classes_{zz}.csv')
            
                with open(f'train_{zz}/w_{zz}.txt', 'w') as file:
                    file.write(str(weights) + '\n' + str(UP) + '\n' + str(DOWN) + '\n' + str(distance_previous))
                    
                oldpwd = os.getcwd()
                os.chdir(f"train_{zz}/")
                
                files = glob.glob("*.csv")
                combined = pd.DataFrame()
                for file in files:
                    data = pd.read_csv(file)
                    combined = pd.concat([combined, data])
                combined.drop(combined.columns[0], axis=1, inplace=True)
                combined.to_csv(f'train_{zz}.csv')
            
                os.chdir(oldpwd)
                
                break
                
                
            weights[zz]+=step  
 
def grad(path, t, oth, l, s, weights):
    '''
    Функция осуществляет градиентное уточнение полученного после первоначального приближения вектора весов 

    Параметры
    ----------
    path : str
        путь к обучающей выборке, подаваемой на вход функции select_top
    t : list
        метка выбранного на этапе первоначального приближения целевого класса
    oth : list
        список, содержащий метки оставшихся классов, которые не относятся к целевому
    l : str
        название столбца с метками классов
    s : int
        шаг изменения значений элементов вектора весов, шаг градиентного уточнения
    weights_main : list
        вектор весов, значения которого необходимо изменить с целью получения большего количества 
        отсеченных экземпляров обучающей выборки

    Пример использования:
    wdl.grad('KAHRAMAN_train.csv', ['High'], ['very_low', 'Low', 'Middle'], 'UNS', 1, [455.02247191011236, 552.8764044943821, 273.38202247191015, 518.2921348314607, 2276.179775280899])

    '''
    weights_all = grad_boost(path, t, oth, l, s, 1, weights)

    with open('weights_all==0.txt', 'w') as file:
        file.write(str(weights_all))

    up_element = 0
    for i1 in range(len(weights)):
        up_element_current = int(weights_all[i1][-3])
        if(up_element < up_element_current):
            up_element = up_element_current
            
    for i3 in range(len(weights)):
        if(int(weights_all[i3][-3]) == up_element):
            weights = weights_all[i3][0]
            up_main = up_element
            down_main = int(weights_all[i3][-2])
            distance_main = weights_all[i3][-1]
    
    print()
    print('ИТОГ = \n')
    print(weights)

    with open('weights==0.txt', 'w') as file:
        file.write(str(weights) + '\nUP = ' + str(up_main) + '\nDOWN = ' + str(down_main) + '\nDistance = ' + str(distance_main))

    print('++++++++++++++++++++++++++++++++++++++')
    print()

    ii = 1
    while True:
        #os.mkdir(f'grad_{ii}')
        weights_all = grad_boost(path, t, oth, l, s, 0, weights)
    
        with open(f'weights_all=={ii}.txt', 'w') as file:
            file.write(str(weights_all))

        up_element = 0
        for i1 in range(len(weights)):
            up_element_current = int(weights_all[i1][-3])
            if(up_element < up_element_current):
                up_element = up_element_current
        
        if(up_element == up_main):##############################################
            down_element = 0
            for i2 in range(len(weights)):
                down_element_current = int(weights_all[i2][-2])
                if(down_element < down_element_current):
                    down_element = down_element_current
            
            if(down_element == down_main):##############################################
                distance_element = 0
                for i4 in range(len(weights)):
                    distance_element_current = weights_all[i4][-1]
                    if(distance_element > distance_element_current):
                        distance_element = distance_element_current
                for i5 in range(len(weights)):
                    if(weights_all[i5][-1] == distance_element):
                        weights = weights_all[i5][0]
                        up_main = int(weights_all[i5][-3])
                        down_main = int(weights_all[i5][-2])
                        distance_main = weights_all[i5][-1] 
                with open(f'weights=={ii}.txt', 'w') as file:
                    file.write(str(weights) + '\nUP = ' + str(up_main) + '\nDOWN = ' + str(down_main) + '\nDistance = ' + str(distance_main))
            
            
                grad_boost_save_files(path, t, oth, l, s, 0, weights)
            
                break
            
            else:
                for i3 in range(len(weights)):
                    if(int(weights_all[i3][-2]) == down_element):
                        weights = weights_all[i3][0]
                        up_main = int(weights_all[i3][-3])
                        down_main = int(weights_all[i3][-2])
                        distance_main = weights_all[i3][-1]           
        else:
            for i3 in range(len(weights)):
                if(int(weights_all[i3][-3]) == up_element):
                    weights = weights_all[i3][0]
                    up_main = int(weights_all[i3][-3])#up_element
                    down_main = int(weights_all[i3][-2])
                    distance_main = weights_all[i3][-1]
    
        print()
        print('ИТОГ = \n')
        print(weights)

        with open(f'weights=={ii}.txt', 'w') as file:
            file.write(str(weights) + '\nUP = ' + str(up_main) + '\nDOWN = ' + str(down_main) + '\nDistance = ' + str(distance_main))

        print('++++++++++++++++++++++++++++++++++++++')
        print()
        ii += 1    

def data_int(path, decimal, name):
    '''
    Функция преобразует набор данных, представленный в вещественнозначном виде в целочисленный вид

    Параметры
    ----------
    path : str
        путь к обучающей выборке в формате csv
    decimal : str
        путь к текстовому файлу, содержащему в каждой строке число, обозначающее количество десятичных знаков после 
        запятой в соответствующем столбце выборки
    name : str
        аргумент функции, обозначающий желаемое название выходных генерируемых файлов

    Пример использования:
    wdl.data_int('kahraman.csv', 'dec.txt', 'KAHRAMAN') 

    '''
    data = pd.read_csv(path) 

    decimal_places = open(decimal, 'r')

    dec = []

    for i in decimal_places:
        dec.append(int(i))

    dec.insert(0, 0)    

    cl = data.columns

    for j in range(1, len(cl)):
        data[cl[j]] = data[cl[j]]*(10**dec[j])

    data_max = data.max()
    data_min = data.min()

    data_max = list(data_max)
    data_max.pop(0)
    data_max = [int(x) for x in data_max]
    data_min = list(data_min)
    data_min.pop(0)
    data_min = [int(x) for x in data_min]


    data_range = []
    for k in range(len(data_max)):
        data_range.append(data_max[k]-data_min[k])

    max_value_range = max(data_range)
    max_index_range = data_range.index(max_value_range)

    lst_data = []
    for t in range(len(data_max)):
        lst_data.append((data_max[max_index_range] - data_min[max_index_range])/(data_max[t] - data_min[t]))

    lst_round = []
    for tt in range(len(data_max)):
        lst_round.append(m.floor((data_max[tt] + data_min[tt])/2))

    lst_data1 = []
    for t1 in range(len(data_max)):
        lst_data1.append(m.floor((data_max[max_index_range] - data_min[max_index_range])/(data_max[t1] - data_min[t1])))

    lst_round.insert(0, 0) 
    lst_data1.insert(0, 0) 
    for y in range(1, len(cl)):
        data[cl[y]] = (data[cl[y]]-lst_round[y])*lst_data1[y]
    
    for z in range(1, len(cl)):
        data[cl[z]] = data[cl[z]].astype(int)
    
    data['N']=data.index
    
    data.to_csv(f'{name}_int.csv')
    
    data_test = data.sample(frac=0.1, random_state=2)
    data_test.to_csv(f'{name}_test.csv')
    
    lstm = []
    for i in data_test['N']:
        lstm.append(i)
    
    data_train = data.drop(labels = lstm,axis = 0)
    data_train.to_csv(f'{name}_train.csv')
    
    return data

def scale_weights(label, data_folder, weights_folder):
    '''
    Функция для масштабирования полученных векторов весов в диапазон от -1 до +1

    Параметры
    ----------
    label : str
        название столбца с метками классов
    data_folder : str
        путь к папке, содержащей обучающие выборки каждого нейрона
    weights_folder : str
        путь к папке, содержащей веса каждого нейрона

    Пример использования:
    wdl.scale_weights('UNS', 'data/', 'weights/')

    '''
    os.mkdir('scale')
    os.mkdir('result')
    os.mkdir('all_weights')
    
    w = []

    # Сортируем список файлов в папке по алфавиту
    files1 = sorted(os.listdir(weights_folder))
    files2 = sorted(os.listdir(data_folder))

    for i in range(1, len(files1)+1):
        filename = f"w_{i}.txt"

    #for file_name in files1:#os.listdir(weights_folder):
    #    if file_name.endswith('.txt'):
        file_path = os.path.join(weights_folder, filename)
            #print()
        with open(file_path, 'r') as file:
            lines = file.readlines()
            w.append(lines[0])           
                
    #print(w)

    weights = []

    for i in range(len(w)):
        string = w[i].replace("'", "").replace("\n", "")
        string1 = string.split(',')

        ww = []

        for j in range(len(string1)):
            z = string1[j].replace("'", "").replace("\n", "").replace("[", "").replace("]", "")
            ww.append(float(z))

        weights.append(ww)

    ##print(weights)

    ii = 0

    for i1 in range(1, len(files2)+1):
        filename = f"train_{i1}.csv"
        file_path = os.path.join(data_folder, filename)
        data = pd.read_csv(file_path)

        print('I = ', ii)
            
        # Рабочий ход
        data['Scalar_calc'] = ""
        #data['Scaled_prod'] = ""
        cl = data.columns
        columns = []
        for j in range(2, len(cl)):
            if(cl[j]=='N'):
                break
            else:
                columns.append(cl[j])
            
        if(len(columns) == 0):
            data = pd.read_csv(file_path, sep = ';')
            data['Scalar_calc'] = ""
            cl = data.columns
            columns = []
            for j in range(2, len(cl)):
                if(cl[j]=='N'):
                    break
                else:
                    columns.append(cl[j])
                        
        print(columns)
        #print()
            
        for i in range(len(data)):
            xx = data.iloc[i]
            x = []
            for u in range(len(columns)):
                x.append(xx[columns[u]])
            f = np.array(weights[ii])
            g = np.array(x)
            scalar = np.dot(f, g)
            data['Scalar_calc'][i]=scalar
        data = data.sort_values(by='Scalar_calc', ascending=False)
        data.to_csv('data.csv')
        data = pd.read_csv('data.csv')
        data.drop(data.columns[0], axis=1, inplace=True)
        firstindex = data[label][0]
        #f'firstindex{ii}' = data[label][0]
        print(data.head())
        print('LENGTH_data = ', len(data))
        print('LABEL = ', firstindex)#data[label][0])#f'firstindex{ii}')
        #print('++++++++++++++++++++++++++++\n')
            
        up = 0
        for i1 in range(len(data)):
            if(data[label][i1]==firstindex):
                up += 1
            else:
                break
            
        li = len(data[label])-1
            
        lastindex = data[label][li]
            
        down = 0
        for i2 in range(len(data)-1, 0, -1):
            if(data[label][i2]==lastindex):
                down += 1
            else:
                break
                    
        data_down = data.sort_values(by='Scalar_calc')
        data_down.to_csv('down.csv')
        data_down = pd.read_csv('down.csv')
            
        #for l in range(down):
        #    d_down = data_down['Scalar_calc'][l]
            
        print('UP = ', up)
        print('DOWN = ', down)
            
        if(len(data) == (up + down)):
            u1 = data['Scalar_calc'][up-1]
            u2 = data_down['Scalar_calc'][down-1]
        else:
            u1 = (data['Scalar_calc'][up]+data['Scalar_calc'][up-1])/2
            u2 = (data_down['Scalar_calc'][down]+data_down['Scalar_calc'][down-1])/2
        print(u1)
        print(u2)
        
        #os.mkdir('scale')
        #os.mkdir('result')
            
        #print(weights[ii])
        #scale = []
        scaled_w = []
        for k in range(len(weights[ii])):
            w1 = 2*weights[ii][k]/(u1 - u2)
            scaled_w.append(w1)
        w2 = -2*u2/(u1-u2)-1
        scaled_w.append(w2)
            #scale.append(scaled_w)
        print(scaled_w)
            
        with open(f'scale/sw_{ii}.txt', 'w') as ff:
            ff.write(str(scaled_w))
            
        data.drop(data.columns[0], axis=1, inplace=True)
                
        data.to_csv(f'result/data_res_{ii}.csv')
            
        print('++++++++++++++++++++++++++++\n')
            
        
        ii += 1
        
    ######################################
    scale = 'scale'
    w = []

    files1 = sorted(os.listdir(scale))

    for i in range(len(files1)):
        filename = f"sw_{i}.txt"
        file_path = os.path.join(scale, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            w.append(lines[0]) 
    
        weights = []

    for i in range(len(w)):
        string = w[i].replace("'", "").replace("\n", "")
        string1 = string.split(',')

        ww = []

        for j in range(len(string1)):
            z = string1[j].replace("'", "").replace("\n", "").replace("[", "").replace("]", "")
            ww.append(float(z))

        weights.append(ww)
        
    formatted_data = []
    for row in weights:
        formatted_row = [str(element) + ',' for element in row]
        formatted_data.append(formatted_row)

    result = '\n'.join([', '.join(row) for row in formatted_data])

    with open("all_weights/weights.txt", "w") as file:
        file.write(result)

    with open("all_weights/weights.txt", "r") as f:
        content = f.read().replace(',,', ',')

    with open("all_weights/weights.txt", "w") as f:
        f.write(content)
    
    return scaled_w

def generate_fa(label, data_folder, weights_folder):
    '''
    Функция, которая на основании проведенного обучения модифицированного полносвязного слоя, генерирует вложенную 
    логическую функцию активации для проверки весовых коэффициентов на тестовых данных

    Параметры
    ----------
    label : str
        название столбца обучающих выборок, в котором содержатся наименования (метки) классов
    data_folder : str
        путь к папке result, сгенерированной в результате выполнения функции scale_weights
    weights_folder : str
        путь к каталогу scale, который сгенерирован после применения функции scale_weights

    Пример использования:
    wdl.generate_fa('UNS', 'result', 'scale')

    '''
    w = []

    files1 = sorted(os.listdir(weights_folder))
    files2 = sorted(os.listdir(data_folder))

    for i in range(0, len(files1)):
        filename = f"sw_{i}.txt"
        file_path = os.path.join(weights_folder, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            w.append(lines[0])           

    weights = []

    for i in range(len(w)):
        string = w[i].replace("'", "").replace("\n", "")
        string1 = string.split(',')

        ww = []

        for j in range(len(string1)):
            z = string1[j].replace("'", "").replace("\n", "").replace("[", "").replace("]", "")
            ww.append(float(z))

        weights.append(ww)

    print('        for n in range(len(data)):')

    ii = 0

    for i1 in range(0, len(files2)):
        filename = f"data_res_{i1}.csv"
        file_path = os.path.join(data_folder, filename)
        data = pd.read_csv(file_path)
            
        data['Scaled_prod'] = ""
            
        cl = data.columns
        columns = []
        for j in range(2, len(cl)):
            if(cl[j]=='N'):
                break
            else:
                columns.append(cl[j])

        for i in range(len(data)):
            xx = data.iloc[i]
            x = []
            for u in range(len(columns)):
                x.append(xx[columns[u]])
            #############
            x.append(1)
            #############
            f = np.array(weights[ii])
            g = np.array(x)
            scalar = np.dot(f, g)
            data['Scaled_prod'][i]=scalar
            
        firstindex = data[label][0]
            
        up = 0
        for i1 in range(len(data)):
            if(data[label][i1]==firstindex):
                up += 1
            else:
                break
                    
        li = len(data[label])-1
            
        lastindex = data[label][li]
            
        down = 0
        for i2 in range(len(data)-1, 0, -1):
            if(data[label][i2]==lastindex):
                down += 1
            else:
                break
                    
        data_down = data.sort_values(by='Scalar_calc')
        data_down.to_csv('down.csv')
        data_down = pd.read_csv('down.csv')
            
        t_up = data['Scaled_prod'][up-1]
        t_down = data_down['Scaled_prod'][down-1]
            
        if(ii == 0):
            if(round(t_up) >= 1):
                print(f'            if(data["SC{ii}"][n]>=1):\n                data["FA"][n]="{firstindex}"')
            if(round(t_down) <= -1):
                print(f'            elif(data["SC{ii}"][n]<=-1):\n                data["FA"][n]="{lastindex}"')
        else:
            if(round(t_up) >= 1):
                print(f'            elif(data["SC{ii}"][n]>=1):\n                data["FA"][n]="{firstindex}"')
            if(round(t_down) <= -1):
                print(f'            elif(data["SC{ii}"][n]<=-1):\n                data["FA"][n]="{lastindex}"')
            
                
        ii += 1

def check_test(train, weights, test, label):
    '''
    Функция осуществляет проверку обучения с помощью правил логического вывода

    Параметры
    ----------
    train : str
        путь к полной целочисленной обучающей выборке
    weights : str
        путь к текстовому файлу weights.txt из сгенерированного каталога all_weights после функции scale_weights
    test : str
        путь к целочисленной тестовой выборке (в случае проверки обучающей выборки передается путь к полной обучающей выборке, 
        аналогичный параметру train)
    label : str
        название столбца с метками классов

    Пример использования:
    wdl.check_test('train_1.csv', 'weights.txt', 'KAHRAMAN_test.csv', 'UNS')

    '''
    def getNameColumn(nameFileTrain):
        '''
        Вспомогательная функция для определения названий всех столбцов, кроме столбца с метками классов

        Параметры
        ----------
        nameFileTrain : str
            путь к обучающей выборке 

        '''
        with open(nameFileTrain, encoding='utf-8') as r_file:
            # Создаем объект DictReader, указываем символ-разделитель ','
            file_reader = csv.DictReader(r_file, delimiter = ',')
            nameColumn  = list(file_reader.fieldnames)  #названия столбцов
        r_file.close()
        del nameColumn[-1]
        del nameColumn[:2]
        return nameColumn
    def getCountNeuronSizeVector(nameFileTrain):
        '''
        Вспомогательная функция для определения необходимого количества нейронов в слое

        Параметры
        ----------
        nameFileTrain : str
            путь к обучающей выборке 

        '''
        with open(nameFileTrain, encoding='utf-8') as r_file:
            # Создаем объект DictReader, указываем символ-разделитель ','
            file_reader = csv.DictReader(r_file, delimiter=',')
            nameColumn  = list(file_reader.fieldnames)  #названия столбцов
            sizeVector = len(nameColumn) - 1          #количество столбцов
            countNeurons = 1
            for row in file_reader:
                countNeurons += 1
        r_file.close()
        return countNeurons, sizeVector
    def initializingDictionaryKeys():
        '''
        Вспомогательная функция для инициализации ключей для правила логического вывода

        '''
        iNeuron = 1                             #инициализация
        while iNeuron <= countNeurons:          #ключей
            rulesLogicalInference[iNeuron] = 0  #для
            rulesLogicalInference[-iNeuron] = 0 #правила
            iNeuron += 1                        #логического
        rulesLogicalInference[iNeuron * 10] = 0 #вывода
    def initializationDictionaryValues(nameFileTrain):
        '''
        Вспомогательная функция для создания словаря логического вывода

        Параметры
        ----------
        nameFileTrain : str
            путь к обучающей выборке 

        '''
        with open(nameFileTrain, encoding='utf-8') as r_file:
            # Создаем объект DictReader, указываем символ-разделитель ','
            file_reader = csv.DictReader(r_file, delimiter=',')
            nameCol = list(file_reader.fieldnames)  # названия столбцов
            for row in file_reader:
                iVector = 0
                for iColumn in nameColumn:  # чтение вектора входов из обучающей выборки
                    qq = row[iColumn]
                    vectorArgs[iVector] = qq
                    iVector += 1
                vectorArgs[iVector] = 1.  # инициализация смещения единицей
                ww = 0.0
                iNeuron = 0
                while iNeuron < countNeurons:               #присвоение
                    ww = np.dot(vectorArgs, matrixWeights[iNeuron])
                    if (ww < -1.) or (ww > 1.):             #значений
                        ww /= abs(ww)                       #соответствующих
                        ww *= (iNeuron + 1)                 #ключам
                        break                               #логического
                    iNeuron += 1                            #
                else:                                       #
                    ww = (countNeurons + 1) * 10            #
                rulesLogicalInference[ww] = row[nameClass]  #вывода
        r_file.close()

    #НАЧАЛО ПРОГРАММЫ
    nameFileTrain = train
    nameFileWeights = weights
    nameFileTest = test
    nameClass = label
    neuronsVectors = getCountNeuronSizeVector(nameFileWeights)
    countNeurons = neuronsVectors[0]
    sizeVector = neuronsVectors[1]
    vectorArgs = np.zeros(sizeVector, dtype=np.float64)
    matrixWeights = np.loadtxt(nameFileWeights, delimiter=',', dtype=np.float64, usecols=range(sizeVector))
    nameColumn = getNameColumn(nameFileTrain)
    rulesLogicalInference = {}
    initializingDictionaryKeys()
    initializationDictionaryValues(nameFileTrain)
    
    ####
    zz = []
    ####
    with open(nameFileTest, encoding='utf-8') as r_file:
        # Создаем объект DictReader, указываем символ-разделитель ','
        file_reader = csv.DictReader(r_file, delimiter=',')
        nameCol = list(file_reader.fieldnames)  # названия столбцов
        for row in file_reader:
            iVector = 0
            for iColumn in nameColumn:  # чтение вектора входов из тестовой выборки
                qq = row[iColumn]
                vectorArgs[iVector] = qq
                iVector += 1
            vectorArgs[iVector] = 1.  # инициализация смещения единицей
            ww = 0.0
            iNeuron = 0
            while iNeuron < countNeurons:  # присвоение
                ww = np.dot(vectorArgs, matrixWeights[iNeuron])
                if (ww < -1.) or (ww > 1.):  # значений
                    ww /= abs(ww)  # соответствующих
                    ww *= (iNeuron + 1)  # ключам
                    break  # логического
                iNeuron += 1  # вывода
            else:  #
                ww = (countNeurons + 1) * 10  #
            if rulesLogicalInference[ww] != row[nameClass]:
                #print(row[''])
                zz.append(row[''])
    
    sizeVector += 1
    
    with open(test,"r", encoding='UTF8') as f:
        reader = csv.reader(f)
        data = list(reader)
        row_count = len(data) - 1
    #print(row_count)
    
    print('Ошибок = ', len(zz), 'из ', row_count)
    print('Экземпляры №:', zz)

def count_conv_operations(height, width, color_channels, kernel_size, stride, num_kernels):
    '''
    Функция для подсчета количества вычислительных операций в сверточном слое 

    Параметры
    ----------
    height : int
        высота исходного изображения выборки данных
    width : int
        ширина исходного изображения
    color_channels : int
        количество цветовых каналов изображения 
        (1 – если изображение представлено в черно-белом одноканальном виде, 3 – если изображение является цветным)
    kernel_size : int
        размерность сверточного ядра квадратной формы
    stride : int
        шаг прохода сверточного ядра по изображению
    num_kernels : int
        количество сверточных ядер в слое

    Пример использования:
    wdl.count_conv_operations(28, 28, 1, 14, 7, 4)

    '''
    output_height = int((height - kernel_size) / stride) + 1
    output_width = int((width - kernel_size) / stride) + 1
    num_operations = (output_height * output_width * kernel_size * kernel_size * num_kernels * 2)*color_channels
    
    no_bias = (num_operations - ((kernel_size**2)*num_kernels))*color_channels
    print('Кол-во вычислительных операций = ', num_operations)  
    print('Кол-во вычислительных операций (no_bias) = ', no_bias)

def auto_select(neuron, path, label, target, quantity):
    '''
    Функция для расчета первоначального приближения с автоматическим выбором целевого класса.

    Параметры
    ----------
    neuron : int
        номер нейрона, для которого производится расчет
    path : str
        путь к целочисленной обучающей выборке в формате csv 
    label : str
        название столбца с метками классов
    target : str, list
        параметр, на основании которого выбирается целевой класс
        1. 'up' - максимальное количество отсеченных экземпляров сверху
        2. 'mx_ratio' - максимальная доля отсеченных экземпляров к неотсеченным сверху
        3. [class] - собственный выбор целевого класса, вместо class необходимо добавить название класса в виде строки или числа 
    quantity : str
        'many' - если обучающая выборка содержит более двух классов
        'binary' - для бинарного варианта набора данных
        
    Пример использования:
    wdl.auto_select(1, 'KH_train.csv', 'UNS', ['very_low'], 'many') 
    wdl.auto_select(8, 'KH_train.csv', 'UNS', 'mx_ratio', 'binary') 

    '''

    if(quantity == 'many'):
        if(os.path.exists('data') == True):
            pass
        else:
            os.mkdir('data')
            
            
        if(os.path.exists('weights') == True):
            pass
        else:
            os.mkdir('weights')
        
        all_classes = pd.read_csv(path)
        all_classes.drop(all_classes.columns[0], axis=1, inplace=True)
        
        uniq = pd.unique(all_classes[[label]].values.ravel('K'))
        uniq = list(uniq)
        
        up_max = 0
        max_ratio = 0
        
        if((target == 'mx_ratio') or (target == 'up')):
            for o in range(len(uniq)):
                uniq1 = uniq.copy()
                z = o
                top = uniq1[z:z+1]
                uniq1.remove(top[0])
                other = uniq1.copy()
            
                class_top = all_classes[all_classes[label].isin(top)] 
                other_classes = all_classes[all_classes[label].isin(other)] 
            
            
                class_top['Scalar_calc'] = ""
                other_classes['Scalar_calc'] = ""
            
                class_top.to_csv('class_top.csv')
                other_classes.to_csv('other_classes.csv')
            
                class_top = pd.read_csv('class_top.csv')
                other_classes = pd.read_csv('other_classes.csv')
            
                class_top.drop(class_top.columns[0], axis=1, inplace=True)
                other_classes.drop(other_classes.columns[0], axis=1, inplace=True)
            
                cl = other_classes.columns
            
                columns = []
                for j in range(1, len(cl)):
                    if(cl[j]=='N'):
                        break
                    else:
                        columns.append(cl[j])
            
                d = []
                for c1 in range(len(columns)):
                    z = f'd{c1}'
                    d.append(z)
            
                class_top_exp = 1
                other_classes_exp = -(len(class_top)/len(other_classes))
            
                for c2 in range(len(d)):
                    class_top[d[c2]] = ''
                    other_classes[d[c2]] = ''
            
                for i1 in range(len(d)):
                    class_top[d[i1]] = class_top[columns[i1]]*class_top_exp
            
                for i2 in range(len(d)):
                    other_classes[d[i2]] = other_classes[columns[i2]]*other_classes_exp
            
                sm1 = class_top.sum()
                sm2 = other_classes.sum()
            
                weights1 = []
                for u in range(len(d)):
                    weights1.append(sm1[d[u]])
            
                weights2 = []
                for uu in range(len(d)):
                    weights2.append(sm2[d[uu]]) 
               
                weights = [x+y for x, y in zip(weights1, weights2)]
             
                for i in range(len(class_top)):
                    xx = class_top.iloc[i]
                    x = []
                    for u in range(len(columns)):
                        x.append(xx[columns[u]])
                
                    f = np.array(weights)
                    g = np.array(x)
                    scalar = np.dot(f, g)
            
                    class_top['Scalar_calc'][i]=scalar
            
                class_top = class_top.sort_values(by='Scalar_calc', ascending=False)
            
                for j in range(len(other_classes)):
                    xx_other = other_classes.iloc[j]
                    x_other = []
                    for uu in range(len(columns)):
                        x_other.append(xx_other[columns[uu]])
                
                    f1 = np.array(weights)
                    g1 = np.array(x_other)
                    scalar1 = np.dot(f1, g1)
                    other_classes['Scalar_calc'][j]=scalar1
            
                other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)
            
                up = 0
                for i in class_top['Scalar_calc']:
                    if(i > other_classes['Scalar_calc'].max()):
                        up+=1
                    else:
                        break
               
                other_classes.to_csv('other_classes.csv')
                other_classes = pd.read_csv('other_classes.csv')
            
                li = len(other_classes[label]) - 1
                
                lastindex_class_other_classes = other_classes[label][li]
            
                if((len(other_classes[label]))==1):
                  down = 1
                else:
                  down = 0
                  for i in range(len(other_classes)-1, 0, -1):
                    if(other_classes[label][i]==lastindex_class_other_classes):
                        down+=1
                    else:
                        break
            
                other_classes.drop(other_classes.columns[0], axis=1, inplace=True)
            
                sum_up_down = up + down
            
                class_top_cut = class_top.iloc[up:]
                other_classes_cut = other_classes.iloc[:-down]
                
                del class_top_cut['Scalar_calc']
                del other_classes_cut['Scalar_calc']
                
                for i1 in range(len(d)):
                    del class_top_cut[d[i1]] 
                
                for i2 in range(len(d)):
                    del other_classes_cut[d[i2]]
                    
                print('Количество отсеченных сверху = ', up)
                print("Количество НЕотсеченных сверху = ", len(class_top_cut))
                print('Количество отсеченных снизу = ', down)
                print("Количество НЕотсеченных снизу = ", len(other_classes_cut))
                print('===Нижняя категория: ', lastindex_class_other_classes)
                print('СУММА отсеченных сверху и снизу = ', sum_up_down)
                print('TOP - ', top)
                print('OTHERS - ', other)    
                print('+++++++++++++++++++++++++')
                
                if(target == 'up'):
                    if(up_max < up):
                        up_max = up
                        top_max = top
                        other_max = other
                        
                        if(os.path.exists('tmp') == False):
                            os.mkdir('tmp')
                        os.mkdir(f'tmp/train_{top_max}{other_max}')
            
                        class_top_cut.to_csv(f'tmp/train_{top_max}{other_max}/class_top_{top_max}{other_max}.csv')
                        other_classes_cut.to_csv(f'tmp/train_{top_max}{other_max}/other_classes_{top_max}{other_max}.csv')
                
                        with open(f'weights/w_{neuron}.txt', 'w') as file:
                            file.write(str(weights))
            
                        folder_path = f'tmp/train_{top_max}{other_max}'
                        file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
                
                if(target == 'mx_ratio'):
                    ratio = up / len(class_top_cut)
                    if(max_ratio < ratio):
                        max_ratio = ratio
                        top_max = top
                        other_max = other
                    
                        if(os.path.exists('tmp') == False):
                            os.mkdir('tmp')
                        os.mkdir(f'tmp/train_{top_max}{other_max}')
            
                        class_top_cut.to_csv(f'tmp/train_{top_max}{other_max}/class_top_{top_max}{other_max}.csv')
                        other_classes_cut.to_csv(f'tmp/train_{top_max}{other_max}/other_classes_{top_max}{other_max}.csv')
                
                        with open(f'weights/w_{neuron}.txt', 'w') as file:
                            file.write(str(weights))
            
                        folder_path = f'tmp/train_{top_max}{other_max}'
                        file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
            
            if len(file_list) < 2:
                print("Должно быть как минимум два CSV файла в папке для объединения.")
                exit()
            
            dfs = [pd.read_csv(os.path.join(folder_path, file)) for file in file_list]
            merged_df = pd.concat(dfs)
            merged_df.drop(merged_df.columns[0], axis=1, inplace=True)
            
            nn = neuron + 1
            fldr = 'data'
            output_path = os.path.join(fldr, f'train_{nn}.csv')
            
            merged_df.to_csv(output_path)
            
            dr = 'tmp'
            shutil.rmtree(dr)
            
            os.remove('class_top.csv')
            os.remove('other_classes.csv')
            
            if(neuron == 1):
                train1 = pd.read_csv(path) 
                train1.drop(train1.columns[0], axis=1, inplace=True)
                train1.to_csv('data/train_1.csv')
        
            print('В качестве целевого выбран класс:', top_max)
        
        else:
            uniq1 = uniq.copy()
            top = target
            uniq1.remove(top[0])
            other = uniq1.copy()
            
            class_top = all_classes[all_classes[label].isin(top)] 
            other_classes = all_classes[all_classes[label].isin(other)] 
        
        
            class_top['Scalar_calc'] = ""
            other_classes['Scalar_calc'] = ""
        
            class_top.to_csv('class_top.csv')
            other_classes.to_csv('other_classes.csv')
        
            class_top = pd.read_csv('class_top.csv')
            other_classes = pd.read_csv('other_classes.csv')
        
            class_top.drop(class_top.columns[0], axis=1, inplace=True)
            other_classes.drop(other_classes.columns[0], axis=1, inplace=True)
        
            cl = other_classes.columns
        
            columns = []
            for j in range(1, len(cl)):
                if(cl[j]=='N'):
                    break
                else:
                    columns.append(cl[j])
        
            d = []
            for c1 in range(len(columns)):
                z = f'd{c1}'
                d.append(z)
        
            class_top_exp = 1
            other_classes_exp = -(len(class_top)/len(other_classes))
        
            for c2 in range(len(d)):
                class_top[d[c2]] = ''
                other_classes[d[c2]] = ''
        
            for i1 in range(len(d)):
                class_top[d[i1]] = class_top[columns[i1]]*class_top_exp
        
            for i2 in range(len(d)):
                other_classes[d[i2]] = other_classes[columns[i2]]*other_classes_exp
        
            sm1 = class_top.sum()
            sm2 = other_classes.sum()
        
            weights1 = []
            for u in range(len(d)):
                weights1.append(sm1[d[u]])
        
            weights2 = []
            for uu in range(len(d)):
                weights2.append(sm2[d[uu]]) 
           
            weights = [x+y for x, y in zip(weights1, weights2)]
         
            for i in range(len(class_top)):
                xx = class_top.iloc[i]
                x = []
                for u in range(len(columns)):
                    x.append(xx[columns[u]])
            
                f = np.array(weights)
                g = np.array(x)
                scalar = np.dot(f, g)
        
                class_top['Scalar_calc'][i]=scalar
        
            class_top = class_top.sort_values(by='Scalar_calc', ascending=False)
        
            for j in range(len(other_classes)):
                xx_other = other_classes.iloc[j]
                x_other = []
                for uu in range(len(columns)):
                    x_other.append(xx_other[columns[uu]])
            
                f1 = np.array(weights)
                g1 = np.array(x_other)
                scalar1 = np.dot(f1, g1)
                other_classes['Scalar_calc'][j]=scalar1
        
            other_classes = other_classes.sort_values(by='Scalar_calc', ascending=False)
        
            up = 0
            for i in class_top['Scalar_calc']:
                if(i > other_classes['Scalar_calc'].max()):
                    up+=1
                else:
                    break
           
            other_classes.to_csv('other_classes.csv')
            other_classes = pd.read_csv('other_classes.csv')
        
            li = len(other_classes[label]) - 1
            
            lastindex_class_other_classes = other_classes[label][li]
        
            if((len(other_classes[label]))==1):
              down = 1
            else:
              down = 0
              for i in range(len(other_classes)-1, 0, -1):
                if(other_classes[label][i]==lastindex_class_other_classes):
                    down+=1
                else:
                    break
        
            other_classes.drop(other_classes.columns[0], axis=1, inplace=True)
        
            sum_up_down = up + down
        
            class_top_cut = class_top.iloc[up:]
            other_classes_cut = other_classes.iloc[:-down]
            
            del class_top_cut['Scalar_calc']
            del other_classes_cut['Scalar_calc']
            
            for i1 in range(len(d)):
                del class_top_cut[d[i1]] 
            
            for i2 in range(len(d)):
                del other_classes_cut[d[i2]]
                
            print('Количество отсеченных сверху = ', up)
            print("Количество НЕотсеченных сверху = ", len(class_top_cut))
            print('Количество отсеченных снизу = ', down)
            print("Количество НЕотсеченных снизу = ", len(other_classes_cut))
            print('===Нижняя категория: ', lastindex_class_other_classes)
            print('СУММА отсеченных сверху и снизу = ', sum_up_down)
            print('TOP - ', top)
            print('OTHERS - ', other)    
            print('+++++++++++++++++++++++++')
        
                
            if(os.path.exists('tmp') == False):
                os.mkdir('tmp')
            os.mkdir(f'tmp/train_{top}{other}')
        
            class_top_cut.to_csv(f'tmp/train_{top}{other}/class_top_{top}{other}.csv')
            other_classes_cut.to_csv(f'tmp/train_{top}{other}/other_classes_{top}{other}.csv')
            
            with open(f'weights/w_{neuron}.txt', 'w') as file:
                file.write(str(weights))
        
            folder_path = f'tmp/train_{top}{other}'
            file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
        
            if len(file_list) < 2:
                print("Должно быть как минимум два CSV файла в папке для объединения.")
                exit()
        
            dfs = [pd.read_csv(os.path.join(folder_path, file)) for file in file_list]
            merged_df = pd.concat(dfs)
            merged_df.drop(merged_df.columns[0], axis=1, inplace=True)
        
            nn = neuron + 1
            fldr = 'data'
            output_path = os.path.join(fldr, f'train_{nn}.csv')
        
            merged_df.to_csv(output_path)
        
            dr = 'tmp'
            shutil.rmtree(dr)
        
            os.remove('class_top.csv')
            os.remove('other_classes.csv')
        
            if(neuron == 1):
                train1 = pd.read_csv(path) 
                train1.drop(train1.columns[0], axis=1, inplace=True)
                train1.to_csv('data/train_1.csv')        
    if(quantity == 'binary'):  
        if(os.path.exists('data') == True):
            pass
        else:
            os.mkdir('data')
            
            
        if(os.path.exists('weights') == True):
            pass
        else:
            os.mkdir('weights')
        
        all_classes_main = pd.read_csv(path)
        all_classes_main.drop(all_classes_main.columns[0], axis=1, inplace=True)
        
        uniq = pd.unique(all_classes_main[[label]].values.ravel('K'))
        uniq = list(uniq)
        
        up_max = 0
        max_ratio = 0
        
        if((target == 'mx_ratio') or (target == 'up')):
            for o in range(len(uniq)):
                uniq1 = uniq.copy()
                z = o
                top = uniq1[z:z+1]
                uniq1.remove(top[0])
                other = uniq1.copy()
            
                class_top = all_classes_main[all_classes_main[label].isin(top)] 
                other_classes = all_classes_main[all_classes_main[label].isin(other)] 
            
                class_top['Scalar_calc'] = ""
                other_classes['Scalar_calc'] = ""
            
                class_top.to_csv('class_top.csv')
                other_classes.to_csv('other_classes.csv')
            
                class_top = pd.read_csv('class_top.csv')
                other_classes = pd.read_csv('other_classes.csv')
            
                class_top.drop(class_top.columns[0], axis=1, inplace=True)
                other_classes.drop(other_classes.columns[0], axis=1, inplace=True)
            
                cl = all_classes_main.columns
            
                columns = []
                for j in range(1, len(cl)):
                    if(cl[j]=='N'):
                        break
                    else:
                        columns.append(cl[j])
            
                d = []
                for c1 in range(len(columns)):
                    z = f'd{c1}'
                    d.append(z)
            
                class_top_exp = 1
                other_classes_exp = 1-(len(class_top)/len(other_classes))
            
                for c2 in range(len(d)):
                    class_top[d[c2]] = ''
                    other_classes[d[c2]] = ''
                
                for i1 in range(len(d)):
                    class_top[d[i1]] = class_top[columns[i1]]*class_top_exp
            
                for i2 in range(len(d)):
                    other_classes[d[i2]] = other_classes[columns[i2]]*other_classes_exp
            
                sm1 = class_top.sum()
                sm2 = other_classes.sum()
            
                weights1 = []
                for u in range(len(d)):
                    weights1.append(sm1[d[u]])
            
                weights2 = []
                for uu in range(len(d)):
                    weights2.append(sm2[d[uu]]) 
                
                weights = [x+y for x, y in zip(weights1, weights2)]
            
                for i in range(len(class_top)):
                    xx = class_top.iloc[i]
                    x = []
                    for u in range(len(columns)):
                        x.append(xx[columns[u]])
                
                    f = np.array(weights)
                    g = np.array(x)
                    scalar = np.dot(f, g)
            
                    class_top['Scalar_calc'][i]=scalar
            
                for j in range(len(other_classes)):
                    xx_other = other_classes.iloc[j]
                    x_other = []
                    for uu in range(len(columns)):
                        x_other.append(xx_other[columns[uu]])
                
                    f1 = np.array(weights)
                    g1 = np.array(x_other)
                    scalar1 = np.dot(f1, g1)
                    other_classes['Scalar_calc'][j]=scalar1
            
                all_classes = pd.concat([class_top, other_classes])
            
                all_classes = all_classes.sort_values(by='Scalar_calc', ascending=False)
                all_classes.to_csv('all_classes.csv')
                all_classes = pd.read_csv('all_classes.csv')
            
                firstindex = all_classes[label][0]
            
                up = 0
                for i in range(len(all_classes)):
                    if(all_classes[label][i]==firstindex):
                        up+=1
                    else:
                        break
            
                li = len(all_classes[label])-1
                
                lastindex_class_binary_classes = all_classes[label][li]
            
                if((len(all_classes[label]))==1):
                    down = 1
                else:
                    down = 0
                    for i in range(len(all_classes)-1, 0, -1):
                        if(all_classes[label][i]==lastindex_class_binary_classes):
                            down+=1
                        else:
                            break
            
                sum_up_down = up + down
            
                dd1 = all_classes['Scalar_calc'][up-1]
                dd2 = all_classes['Scalar_calc'][up]
                distance = dd1 - dd2
            
                all_classes_upcut = all_classes.iloc[up:]
                all_classes_allcut = all_classes_upcut.iloc[:-down]
                
                del all_classes_allcut['Scalar_calc']
        
                for i1 in range(len(d)):
                    del all_classes_allcut[d[i1]] 
                
                print('Количество отсеченных сверху = ', up)
                print('Количество отсеченных снизу = ', down)
                print('===Верхняя категория: ', firstindex)
                print('===Нижняя категория: ', lastindex_class_binary_classes)
                print('TOP - ', top)
                print('OTHER - ', other) 
                print('СУММА отсеченных сверху и снизу = ', sum_up_down)
                print('+++++++++++++++++++++++++++++++++++++++')
            
                nn = neuron + 1
                if(target == 'up'):
                    if(up_max < up):
                        up_max = up
                        top_max = top
                        other_max = other
                        
                        all_classes_allcut.drop(all_classes_allcut.columns[0], axis=1, inplace=True)
                        all_classes_allcut.to_csv(f'data/train_{nn}.csv')
                    
                        with open(f'weights/w_{neuron}.txt', 'w') as file:
                            file.write(str(weights))
                if(target == 'mx_ratio'):
                    if(firstindex == top[0]):
                        ratio = up / len(class_top)
                    else:
                        ratio = up / len(other_classes)
                    if(max_ratio < ratio):
                        
                        max_ratio = ratio
                        top_max = top
                        other_max = other
            
                        all_classes_allcut.drop(all_classes_allcut.columns[0], axis=1, inplace=True)
                        all_classes_allcut.to_csv(f'data/train_{nn}.csv')
                    
                        with open(f'weights/w_{neuron}.txt', 'w') as file:
                            file.write(str(weights))
                
            print('В качестве целевого выбран класс:', top_max)
            os.remove('class_top.csv')
            os.remove('other_classes.csv')
            os.remove('all_classes.csv')
            
            if(os.path.exists(f'data/train_{neuron}.csv') == False):
                train1 = pd.read_csv(path) 
                train1.drop(train1.columns[0], axis=1, inplace=True)
                train1.to_csv(f'data/train_{neuron}.csv')
        else:
            uniq1 = uniq.copy()
            top = target
            uniq1.remove(top[0])
            other = uniq1.copy()
        
            class_top = all_classes_main[all_classes_main[label].isin(top)] 
            other_classes = all_classes_main[all_classes_main[label].isin(other)] 
        
            class_top['Scalar_calc'] = ""
            other_classes['Scalar_calc'] = ""
        
            class_top.to_csv('class_top.csv')
            other_classes.to_csv('other_classes.csv')
        
            class_top = pd.read_csv('class_top.csv')
            other_classes = pd.read_csv('other_classes.csv')
        
            class_top.drop(class_top.columns[0], axis=1, inplace=True)
            other_classes.drop(other_classes.columns[0], axis=1, inplace=True)
        
            cl = all_classes_main.columns
        
            columns = []
            for j in range(1, len(cl)):
                if(cl[j]=='N'):
                    break
                else:
                    columns.append(cl[j])
        
            d = []
            for c1 in range(len(columns)):
                z = f'd{c1}'
                d.append(z)
        
            class_top_exp = 1
            other_classes_exp = 1-(len(class_top)/len(other_classes))
        
            for c2 in range(len(d)):
                class_top[d[c2]] = ''
                other_classes[d[c2]] = ''
            
            for i1 in range(len(d)):
                class_top[d[i1]] = class_top[columns[i1]]*class_top_exp
        
            for i2 in range(len(d)):
                other_classes[d[i2]] = other_classes[columns[i2]]*other_classes_exp
        
            sm1 = class_top.sum()
            sm2 = other_classes.sum()
        
            weights1 = []
            for u in range(len(d)):
                weights1.append(sm1[d[u]])
        
            weights2 = []
            for uu in range(len(d)):
                weights2.append(sm2[d[uu]]) 
            
            weights = [x+y for x, y in zip(weights1, weights2)]
        
            for i in range(len(class_top)):
                xx = class_top.iloc[i]
                x = []
                for u in range(len(columns)):
                    x.append(xx[columns[u]])
            
                f = np.array(weights)
                g = np.array(x)
                scalar = np.dot(f, g)
        
                class_top['Scalar_calc'][i]=scalar
        
            for j in range(len(other_classes)):
                xx_other = other_classes.iloc[j]
                x_other = []
                for uu in range(len(columns)):
                    x_other.append(xx_other[columns[uu]])
            
                f1 = np.array(weights)
                g1 = np.array(x_other)
                scalar1 = np.dot(f1, g1)
                other_classes['Scalar_calc'][j]=scalar1
        
            all_classes = pd.concat([class_top, other_classes])
        
            all_classes = all_classes.sort_values(by='Scalar_calc', ascending=False)
            all_classes.to_csv('all_classes.csv')
            all_classes = pd.read_csv('all_classes.csv')
        
            firstindex = all_classes[label][0]
        
            up = 0
            for i in range(len(all_classes)):
                if(all_classes[label][i]==firstindex):
                    up+=1
                else:
                    break
        
            li = len(all_classes[label])-1
            
            lastindex_class_binary_classes = all_classes[label][li]
        
            if((len(all_classes[label]))==1):
                down = 1
            else:
                down = 0
                for i in range(len(all_classes)-1, 0, -1):
                    if(all_classes[label][i]==lastindex_class_binary_classes):
                        down+=1
                    else:
                        break
        
            sum_up_down = up + down
        
            dd1 = all_classes['Scalar_calc'][up-1]
            dd2 = all_classes['Scalar_calc'][up]
            distance = dd1 - dd2
        
            all_classes_upcut = all_classes.iloc[up:]
            all_classes_allcut = all_classes_upcut.iloc[:-down]
            
            del all_classes_allcut['Scalar_calc']
        
            for i1 in range(len(d)):
                del all_classes_allcut[d[i1]] 
            print('Количество отсеченных сверху = ', up)
            print('Количество отсеченных снизу = ', down)
            print('===Верхняя категория: ', firstindex)
            print('===Нижняя категория: ', lastindex_class_binary_classes)
            print('TOP - ', top)
            print('OTHER - ', other) 
            print('СУММА отсеченных сверху и снизу = ', sum_up_down)
            print('+++++++++++++++++++++++++++++++++++++++')
            
            nn = neuron + 1
            
            all_classes_allcut.drop(all_classes_allcut.columns[0], axis=1, inplace=True)
            all_classes_allcut.to_csv(f'data/train_{nn}.csv')
                    
            with open(f'weights/w_{neuron}.txt', 'w') as file:
                file.write(str(weights))
        
            os.remove('class_top.csv')
            os.remove('other_classes.csv')
            os.remove('all_classes.csv')
            
            if(os.path.exists(f'data/train_{neuron}.csv') == False):
                train1 = pd.read_csv(path) 
                train1.drop(train1.columns[0], axis=1, inplace=True)
                train1.to_csv(f'data/train_{neuron}.csv')


        

