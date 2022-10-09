
import pandas as pd
import numpy as np

class_up = pd.read_csv('final/class_up.csv')
class_down = pd.read_csv('final/class_down.csv')
class_middle = pd.read_csv('final/class_middle.csv')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('max_seq_items', None)

class_up.drop(class_up.columns[0], axis=1, inplace=True)  
class_down.drop(class_down.columns[0], axis=1, inplace=True)   
class_middle.drop(class_middle.columns[0], axis=1, inplace=True) 

class_up.drop(class_up.columns[0], axis=1, inplace=True)  
class_down.drop(class_down.columns[0], axis=1, inplace=True)   
class_middle.drop(class_middle.columns[0], axis=1, inplace=True) 

class_up.drop(class_up.columns[0], axis=1, inplace=True)  
class_down.drop(class_down.columns[0], axis=1, inplace=True)   
class_middle.drop(class_middle.columns[0], axis=1, inplace=True) 

Alcohol = 0
for i in range(len(class_up)):
    Alcohol += class_up['d0'][i]
        
Malic_acid = 0
for i in range(len(class_up)):
    Malic_acid += class_up['d1'][i]
        
Ash = 0
for i in range(len(class_up)):
    Ash += class_up['d2'][i]
        
Acl = 0
for i in range(len(class_up)):
    Acl += class_up['d3'][i]
        
Mg = 0
for i in range(len(class_up)):
    Mg += class_up['d4'][i]
    
Phenols = 0
for i in range(len(class_up)):
    Phenols += class_up['d5'][i]
        
Flavanoids = 0
for i in range(len(class_up)):
    Flavanoids += class_up['d6'][i]
        
Nonflavanoid_phenols = 0
for i in range(len(class_up)):
    Nonflavanoid_phenols += class_up['d7'][i]
        
Proanth = 0
for i in range(len(class_up)):
    Proanth += class_up['d8'][i]
        
Color_int = 0
for i in range(len(class_up)):
    Color_int += class_up['d9'][i]
        
Hue = 0
for i in range(len(class_up)):
    Hue += class_up['d10'][i]
        
OD = 0
for i in range(len(class_up)):
    OD += class_up['d11'][i]
        
Proline = 0
for i in range(len(class_up)):
    Proline += class_up['d12'][i]
    
    
weights_up = [Alcohol, 
                  Malic_acid, Ash, Acl, Mg, Phenols, Flavanoids, Nonflavanoid_phenols, Proanth, Color_int, Hue, OD, Proline]

sum_cut_safe = 59
sum_distance_safe = 1.23610471486516

learning_rate=1

infinity_distance = (class_down['Distance_to_Borders'].max())*2

for j in range(len(class_down)): 
    # 1.
    index = np.where(class_down['Distance_to_Borders'] == class_down['Distance_to_Borders'].min())
    index = index[0][0]

    temp = class_down['Distance_to_Borders'].min()
    temp1 = class_down['N'][index]

    # 2.
    class_down['Expect'][index] = -1
    
    
    class_down['d0'][index] = class_down['Expect'][index]*class_down['Alcohol'][index]*learning_rate
    class_down['d1'][index] = class_down['Expect'][index]*class_down['Malic.acid'][index]*learning_rate
    class_down['d2'][index] = class_down['Expect'][index]*class_down['Ash'][index]*learning_rate
    class_down['d3'][index] = class_down['Expect'][index]*class_down['Acl'][index]*learning_rate
    class_down['d4'][index] = class_down['Expect'][index]*class_down['Mg'][index]*learning_rate
    class_down['d5'][index] = class_down['Expect'][index]*class_down['Phenols'][index]*learning_rate
    class_down['d6'][index] = class_down['Expect'][index]*class_down['Flavanoids'][index]*learning_rate
    class_down['d7'][index] = class_down['Expect'][index]*class_down['Nonflavanoid.phenols'][index]*learning_rate
    class_down['d8'][index] = class_down['Expect'][index]*class_down['Proanth'][index]*learning_rate
    class_down['d9'][index] = class_down['Expect'][index]*class_down['Color.int'][index]*learning_rate*learning_rate
    class_down['d10'][index] = class_down['Expect'][index]*class_down['Hue'][index]*learning_rate
    class_down['d11'][index] = class_down['Expect'][index]*class_down['OD'][index]*learning_rate
    class_down['d12'][index] = class_down['Expect'][index]*class_down['Proline'][index]*learning_rate

    # 3.
    weights_down = [weights_up[0] + class_down['d0'][index], weights_up[1] + class_down['d1'][index],
               weights_up[2] + class_down['d2'][index], weights_up[3] + class_down['d3'][index],
               weights_up[4] + class_down['d4'][index], weights_up[5] + class_down['d5'][index],
               weights_up[6] + class_down['d6'][index], weights_up[7] + class_down['d7'][index],
               weights_up[8] + class_down['d8'][index], weights_up[9] + class_down['d9'][index],
               weights_up[10] + class_down['d10'][index], weights_up[11] + class_down['d11'][index],
               weights_up[12] + class_down['d12'][index]]

    # 4.
    for i in range(len(class_up['Wine'])):
        class_up['Scalar_calc'][i]=weights_down[0]*class_up['Alcohol'][i] + \
        weights_down[1]*class_up['Malic.acid'][i] + weights_down[2]*class_up['Ash'][i] + \
        weights_down[3]*class_up['Acl'][i] + weights_down[4]*class_up['Mg'][i] + \
        weights_down[5]*class_up['Phenols'][i] + weights_down[6]*class_up['Flavanoids'][i] + \
        weights_down[7]*class_up['Nonflavanoid.phenols'][i] + weights_down[8]*class_up['Proanth'][i] + \
        weights_down[9]*class_up['Color.int'][i] + weights_down[10]*class_up['Hue'][i] + \
        weights_down[11]*class_up['OD'][i] + weights_down[12]*class_up['Proline'][i]

    for i in range(len(class_down['Wine'])):
        class_down['Scalar_calc'][i]=weights_down[0]*class_down['Alcohol'][i] + \
        weights_down[1]*class_down['Malic.acid'][i] + weights_down[2]*class_down['Ash'][i] + \
        weights_down[3]*class_down['Acl'][i] + weights_down[4]*class_down['Mg'][i] + \
        weights_down[5]*class_down['Phenols'][i] + weights_down[6]*class_down['Flavanoids'][i] + \
        weights_down[7]*class_down['Nonflavanoid.phenols'][i] + weights_down[8]*class_down['Proanth'][i] + \
        weights_down[9]*class_down['Color.int'][i] + weights_down[10]*class_down['Hue'][i] + \
        weights_down[11]*class_down['OD'][i] + weights_down[12]*class_down['Proline'][i]
        
    for i in range(len(class_middle['Wine'])):
        class_middle['Scalar_calc'][i]=weights_down[0]*class_middle['Alcohol'][i] + \
        weights_down[1]*class_middle['Malic.acid'][i] + weights_down[2]*class_middle['Ash'][i] + \
        weights_down[3]*class_middle['Acl'][i] + weights_down[4]*class_middle['Mg'][i] + \
        weights_down[5]*class_middle['Phenols'][i] + weights_down[6]*class_middle['Flavanoids'][i] + \
        weights_down[7]*class_middle['Nonflavanoid.phenols'][i] + weights_down[8]*class_middle['Proanth'][i] + \
        weights_down[9]*class_middle['Color.int'][i] + weights_down[10]*class_middle['Hue'][i] + \
        weights_down[11]*class_middle['OD'][i] + weights_down[12]*class_middle['Proline'][i]

    # 5.
    class_up = class_up.sort_values(by='Scalar_calc', ascending=False)
    class_down = class_down.sort_values(by='Scalar_calc', ascending=False)
    class_middle = class_middle.sort_values(by='Scalar_calc', ascending=False)

    class_up.to_csv('class_up.csv')
    class_down.to_csv('class_down.csv')
    class_middle.to_csv('class_middle.csv')

    class_up.drop(class_up.columns[0], axis=1, inplace=True)  
    class_down.drop(class_down.columns[0], axis=1, inplace=True)   
    class_middle.drop(class_middle.columns[0], axis=1, inplace=True) 

    class_up = pd.read_csv('class_up.csv')
    class_down = pd.read_csv('class_down.csv')
    class_middle = pd.read_csv('class_middle.csv')

    # 6.
    number_truncate_up = 0
    for i in class_up['Scalar_calc']:
        if(i > class_middle['Scalar_calc'].max()):
            number_truncate_up+=1
        else:
            break

    # 7.
    number_truncate_down = 0
    for i in range(len(class_down['Scalar_calc'])-1, 0, -1):
        if(class_down['Scalar_calc'][i] < class_middle['Scalar_calc'].min()):
            number_truncate_down+=1
        else:
            break

    # 8.
    sum_truncate_current = number_truncate_up + number_truncate_down

    # 9.
    for i in range(number_truncate_up+1):
        distance_up = class_up['Scalar_calc'][i]
    
    class_d2 = class_down.sort_values(by='Scalar_calc')
    class_d2.to_csv('class_d2.csv')
    class_d2 = pd.read_csv('class_d2.csv')
    
    for i in range(number_truncate_down+1):
        distance_down = class_d2['Scalar_calc'][i]
    
    raz1 = class_middle['Scalar_calc'].max() - distance_up
    raz2 = distance_down - class_middle['Scalar_calc'].min()

    # 10.
    sum_distance_current = raz1 + raz2

    # 11.
    z = np.where(class_down['Distance_to_Borders'] == class_down['Distance_to_Borders'].min())
    z = z[0][0]
    class_down['Distance_to_Borders'][z] = infinity_distance

    # 12.
    if(sum_truncate_current < sum_cut_safe):
        class_down['Expect'][z] = 0
        class_down['d0'][z] = 0
        class_down['d1'][z] = 0
        class_down['d2'][z] = 0
        class_down['d3'][z] = 0
        class_down['d4'][z] = 0
        class_down['d5'][z] = 0
        class_down['d6'][z] = 0
        class_down['d7'][z] = 0
        class_down['d8'][z] = 0
        class_down['d9'][z] = 0
        class_down['d10'][z] = 0
        class_down['d11'][z] = 0
        class_down['d12'][z] = 0
    elif(sum_truncate_current > sum_cut_safe):
        sum_cut_safe = sum_truncate_current
        sum_distance_safe = sum_distance_current
        weights_up = weights_down
    elif(sum_distance_current < sum_distance_safe):
        sum_cut_safe = sum_truncate_current
        sum_distance_safe = sum_distance_current
        weights_up = weights_down
    else:
        class_down['Expect'][z] = 0
        class_down['Expect'][z] = 0
        class_down['d0'][z] = 0
        class_down['d1'][z] = 0
        class_down['d2'][z] = 0
        class_down['d3'][z] = 0
        class_down['d4'][z] = 0
        class_down['d5'][z] = 0
        class_down['d6'][z] = 0
        class_down['d7'][z] = 0
        class_down['d8'][z] = 0
        class_down['d9'][z] = 0
        class_down['d10'][z] = 0
        class_down['d11'][z] = 0
        class_down['d12'][z] = 0
        
    class_up.drop(class_up.columns[0], axis=1, inplace=True)  
    class_down.drop(class_down.columns[0], axis=1, inplace=True)   
    class_middle.drop(class_middle.columns[0], axis=1, inplace=True) 
        
    print(temp1)
    print('_________________')
    























