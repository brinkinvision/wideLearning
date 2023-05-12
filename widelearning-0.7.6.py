
import numpy as np
import pandas as pd
import os
import shutil
import glob
import math as m
import requests

def txt_kernel(file_path):	
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
    
# horizontal(32, [-7,-5,-3,-1,1,3,5,7])
def horizontal(dim, values):
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

# vertical(32, [-7,-5,-3,-1,1,3,5,7])	
def vertical(dim, values):
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

# diagonal([-7, -5, -3, -1], [1, 3, 5, 7], 1, 32)	
def diagonal(up, down, d, dimension):

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

#w1 = txt_kernel_rgb('k1.txt', 'k2.txt', 'k3.txt')

def add_kernel_rgb(k1, k2, k3, w):

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

    for i in range(len(w1)):
        for j in range(len(w1)):
            w[i][j][0].append(w11[i][j])
	########################################
    for i in range(len(w1)):
        for j in range(len(w1)):
            w[i][j][1].append(w22[i][j])
	########################################
    for i in range(len(w1)):
        for j in range(len(w1)):
            w[i][j][2].append(w33[i][j])
	########################################
	
    return w

#w2 = add_kernel_rgb('k3.txt', 'k2.txt', 'k1.txt', weights1)
        
def select_top(path, label):
    # label - название столбца с метками классов ('Wine')
    # z - индекс в переборе меток классов
    all_classes = pd.read_csv(path)#('wine_train.csv')
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
        
        #weights_all_categories.append(weights)

def grad_boost(path, t, oth, l, s, from_initial, weights_main):
    # path - путь к обучающей выборке
    # t - метка целевого класса в виде списка
    # oth - список меток оставшхся классов
    # l- название столбца с метками классов ('Wine')
    # s - шаг градиента
    
    # from_initial - 1, если вектор берется из select_top (грубое приближение), 
    # 0 - если вектор снова корректируется (после grad_boost)
    
    # weights_main - весовые коэффиценты в виде списка
    # zz - индекс в переборе элементов вектора весов
    
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

#grad_boost("kahraman_train4.csv", ['high'], ['middle', 'low', 'very_low'], 'UNS', 1, 1, [-47.32524271844662, 358.46601941747576, 480.09708737864077, -221.2135922330097, 260.9077669902913])

def grad_boost_save_files(path, t, oth, l, s, from_initial, weights_main):
    # path - путь к обучающей выборке
    # t - метка целевого класса в виде списка
    # oth - список меток оставшхся классов
    # l- название столбца с метками классов ('Wine')
    # s - шаг градиента
    
    # from_initial - 1, если вектор берется из select_top (грубое приближение), 
    # 0 - если вектор снова корректируется (после grad_boost)
    
    # weights_main - весовые коэффиценты в виде списка
    # zz - индекс в переборе элементов вектора весов
    
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

def grad_boost_binary(path, t, oth, l, s, from_initial, weights_main):
    #path = 'trainW07_1.csv'
    #t = ['low']
    #oth = ['middle']
    #l = 'UNS'
    #s = 1
    #from_initial = 1
    #weights_main = [-663.0, -8850.0, -610.0, -16520.0, -1202.0] 
    
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
 
                
# grad("trainW08_4.csv", ['very_low'], ['high', 'low', 'middle'], 'UNS', [-837.2851063829785, 1654.2510638297872, -485.82978723404267, 1729.021276595745, 19057.587234042556])
def grad(path, t, oth, l, weights):
    #path = "trainW08_4.csv"#"kahraman_train.csv"
    #t = ['high']#['very_low']
    #oth = ['very_low', 'low', 'middle']#['high', 'low', 'middle']
    #l = 'UNS'
    s = 1

    #weights = [-837.2851063829785, 1654.2510638297872, -485.82978723404267, 1729.021276595745, 19057.587234042556]
    #[-4745.458204334365, -5586.578947368422, -5510.185758513931, -8949.30959752322, -18447.66253869969]

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

# data_int('forest.csv', 'dec_forest.txt', 'forest')           
def data_int(path, decimal, name):
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

def generate_fa(data_folder, weights_folder, label):

    w = []

    # Сортируем список файлов в папке по алфавиту
    files1 = sorted(os.listdir(weights_folder))
    files2 = sorted(os.listdir(data_folder))

    for i in range(1, len(files1)+1):
        filename = f"w_{i}.txt"

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

    ii = 0

    for i1 in range(1, len(files2)+1):
        filename = f"train_{i1}.csv"
        file_path = os.path.join(data_folder, filename)
        data = pd.read_csv(file_path)

        print('I = ', ii)
            
        # Рабочий ход
        data['Scalar_calc'] = ""
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
        print(data.head())
        print('LENGTH_data = ', len(data))
        print('LABEL = ', firstindex)#
            
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

        scaled_w = []
        for k in range(len(weights[ii])):
            w1 = 2*weights[ii][k]/(u1 - u2)
            scaled_w.append(w1)
        w2 = -2*u2/(u1-u2)-1
        scaled_w.append(w2)
        print(scaled_w)
            
        with open(f'scale/sw_{ii}.txt', 'w') as ff:
            ff.write(str(scaled_w))
            
        data.drop(data.columns[0], axis=1, inplace=True)
                
        data.to_csv(f'result/data_res_{ii}.csv')
            
        print('++++++++++++++++++++++++++++\n')
            
        
        ii += 1

    ##########################################

    label = 'target'

    data_folder = 'result/'

    weights_folder = 'scale/'

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
            if(t_up >= 1):
                print(f'            if(data["SC{ii}"][n]>=1):\n                data["FA"][n]={firstindex}')
            if(t_down <= -1):
                print(f'            elif(data["SC{ii}"][n]<=-1):\n                data["FA"][n]={lastindex}')
        else:
            if(t_up >= 1):
                print(f'            elif(data["SC{ii}"][n]>=1):\n                data["FA"][n]={firstindex}')
            if(t_down <= -1):
                print(f'            elif(data["SC{ii}"][n]<=-1):\n                data["FA"][n]={lastindex}')
            
                
        ii += 1

    return scaled_w

def check_fa(data_folder, weights_folder, label):

    label = 'target'

    data_folder = 'test/'

    weights_folder = 'scale/'

    w = []

    files1 = sorted(os.listdir(weights_folder))

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

    for file_name in os.listdir(data_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_folder, file_name)
            data = pd.read_csv(file_path)
            
            
            
            for k1 in range(len(weights)):
                data[f'SC{k1}'] = ""
            
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
                    f = np.array(weights[k1])
                    g = np.array(x)
                    scalar = np.dot(f, g)
                    data[f'SC{k1}'][i]=scalar
            
            data['FA']=""

    # Место для вставки сгенерированного кода
            
            for n in range(len(data)):
                if(data["SC0"][n]>=1):
                    data["FA"][n]=1
                elif(data["SC0"][n]<=-1):
                    data["FA"][n]=0
                elif(data["SC1"][n]>=1):
                    data["FA"][n]=2
                elif(data["SC1"][n]<=-1):
                    data["FA"][n]=0
                elif(data["SC2"][n]>=1):
                    data["FA"][n]=0
                elif(data["SC2"][n]<=-1):
                    data["FA"][n]=1
            
    ########################################
            
            mistakes = []
            data['check']=""
            for n1 in range(len(data)):
                if((data['target'][n1]==data['FA'][n1])==True):
                    data['check'][n1]=0
                else:
                    data['check'][n1]=1
                    mistakes.append(data['N'][n1])
            
            data.to_csv('final/data_test.csv')
            print(data.head())
            print('+++++++++++++++++++')
            print('Ошибок: ', data['check'].sum(), '/', len(data))
            print(mistakes)

    return data
        

