
import pandas as pd
import numpy as np

infinity_distance = 10.70325415300089*2

sum_distance_safe = 5.598579771006952

up = 47
down = 6
sum_cut_safe = up + down

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('max_seq_items', None)

class_up = pd.read_csv('cl/class_up.csv')
class_down = pd.read_csv('cl/class_down.csv')
class_middle = pd.read_csv('cl/class_middle.csv')

class_up.drop(class_up.columns[0], axis=1, inplace=True)  
class_down.drop(class_down.columns[0], axis=1, inplace=True)   
class_middle.drop(class_middle.columns[0], axis=1, inplace=True)  

for j in range(len(class_up)):  
    
    index = np.where(class_up['Distance_to_Borders'] == class_up['Distance_to_Borders'].min())
    index = index[0][0]
    temp = class_up['Distance_to_Borders'].min()
    temp1 = class_up['N'][index]
    class_up['Expect'][index] = 0
   
    template_delta = [class_up['d0'][index], 
                 class_up['d1'][index],
                 class_up['d2'][index],
                 class_up['d3'][index],
                 class_up['d4'][index],
                 class_up['d5'][index],
                 class_up['d6'][index],
                 class_up['d7'][index],
                 class_up['d8'][index],
                 class_up['d9'][index],
                 class_up['d10'][index],
                 class_up['d11'][index],
                 class_up['d12'][index]]
  
    class_up['d0'][index] = 0
    class_up['d1'][index] = 0
    class_up['d2'][index] = 0
    class_up['d3'][index] = 0
    class_up['d4'][index] = 0
    class_up['d5'][index] = 0
    class_up['d6'][index] = 0
    class_up['d7'][index] = 0
    class_up['d8'][index] = 0
    class_up['d9'][index] = 0
    class_up['d10'][index] = 0
    class_up['d11'][index] = 0
    class_up['d12'][index] = 0

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
    
    for i in range(len(class_up['Wine'])):
        class_up['Scalar_calc'][i]=weights_up[0]*class_up['Alcohol'][i] + \
        weights_up[1]*class_up['Malic.acid'][i] + weights_up[2]*class_up['Ash'][i] + \
        weights_up[3]*class_up['Acl'][i] + weights_up[4]*class_up['Mg'][i] + \
        weights_up[5]*class_up['Phenols'][i] + weights_up[6]*class_up['Flavanoids'][i] + \
        weights_up[7]*class_up['Nonflavanoid.phenols'][i] + weights_up[8]*class_up['Proanth'][i] + \
        weights_up[9]*class_up['Color.int'][i] + weights_up[10]*class_up['Hue'][i] + \
        weights_up[11]*class_up['OD'][i] + weights_up[12]*class_up['Proline'][i]

    for i in range(len(class_down['Wine'])):
        class_down['Scalar_calc'][i]=weights_up[0]*class_down['Alcohol'][i] + \
        weights_up[1]*class_down['Malic.acid'][i] + weights_up[2]*class_down['Ash'][i] + \
        weights_up[3]*class_down['Acl'][i] + weights_up[4]*class_down['Mg'][i] + \
        weights_up[5]*class_down['Phenols'][i] + weights_up[6]*class_down['Flavanoids'][i] + \
        weights_up[7]*class_down['Nonflavanoid.phenols'][i] + weights_up[8]*class_down['Proanth'][i] + \
        weights_up[9]*class_down['Color.int'][i] + weights_up[10]*class_down['Hue'][i] + \
        weights_up[11]*class_down['OD'][i] + weights_up[12]*class_down['Proline'][i]
        
    for i in range(len(class_middle['Wine'])):
        class_middle['Scalar_calc'][i]=weights_up[0]*class_middle['Alcohol'][i] + \
        weights_up[1]*class_middle['Malic.acid'][i] + weights_up[2]*class_middle['Ash'][i] + \
        weights_up[3]*class_middle['Acl'][i] + weights_up[4]*class_middle['Mg'][i] + \
        weights_up[5]*class_middle['Phenols'][i] + weights_up[6]*class_middle['Flavanoids'][i] + \
        weights_up[7]*class_middle['Nonflavanoid.phenols'][i] + weights_up[8]*class_middle['Proanth'][i] + \
        weights_up[9]*class_middle['Color.int'][i] + weights_up[10]*class_middle['Hue'][i] + \
        weights_up[11]*class_middle['OD'][i] + weights_up[12]*class_middle['Proline'][i]
    
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
    
    number_truncate_up = 0
    for i in class_up['Scalar_calc']:
        if(i > class_middle['Scalar_calc'].max()):
            number_truncate_up+=1
        else:
            break
    
    number_truncate_down = 0
    for i in range(len(class_down['Scalar_calc'])-1, 0, -1):
        if(class_down['Scalar_calc'][i] < class_middle['Scalar_calc'].min()):
            number_truncate_down+=1
        else:
            break
            
    for i in range(number_truncate_up+1):
        distance_up = class_up['Scalar_calc'][i]
    
    class_d2 = class_down.sort_values(by='Scalar_calc')
    class_d2.to_csv('class_d2.csv')
    class_d2 = pd.read_csv('class_d2.csv')
    
    for i in range(number_truncate_down+1):
        distance_down = class_d2['Scalar_calc'][i]
   
    
    sum_truncate_current = number_truncate_up + number_truncate_down
    
    raz1 = class_middle['Scalar_calc'].max() - distance_up
    
    raz2 = distance_down - class_middle['Scalar_calc'].min()
    
    sum_distance_current = raz1 + raz2
    
    z = np.where(class_up['Distance_to_Borders'] == class_up['Distance_to_Borders'].min())
    z = z[0][0]
    class_up['Distance_to_Borders'][z] = infinity_distance
    
    if(sum_truncate_current < sum_cut_safe):
        class_up['Expect'][z] = 1
        class_up['d0'][z] = template_delta[0]
        class_up['d1'][z] = template_delta[1]
        class_up['d2'][z] = template_delta[2]
        class_up['d3'][z] = template_delta[3]
        class_up['d4'][z] = template_delta[4]
        class_up['d5'][z] = template_delta[5]
        class_up['d6'][z] = template_delta[6]
        class_up['d7'][z] = template_delta[7]
        class_up['d8'][z] = template_delta[8]
        class_up['d9'][z] = template_delta[9]
        class_up['d10'][z] = template_delta[10]
        class_up['d11'][z] = template_delta[11]
        class_up['d12'][z] = template_delta[12] 
        
    elif(sum_truncate_current > sum_cut_safe):
        sum_cut_safe = sum_truncate_current
        sum_distance_safe = sum_distance_current
        
    elif(sum_distance_current < sum_distance_safe):
        sum_cut_safe = sum_truncate_current
        sum_distance_safe = sum_distance_current
        
    else:
        class_up['Expect'][z] = 1
        class_up['d0'][z] = template_delta[0]
        class_up['d1'][z] = template_delta[1]
        class_up['d2'][z] = template_delta[2]
        class_up['d3'][z] = template_delta[3]
        class_up['d4'][z] = template_delta[4]
        class_up['d5'][z] = template_delta[5]
        class_up['d6'][z] = template_delta[6]
        class_up['d7'][z] = template_delta[7]
        class_up['d8'][z] = template_delta[8]
        class_up['d9'][z] = template_delta[9]
        class_up['d10'][z] = template_delta[10]
        class_up['d11'][z] = template_delta[11]
        class_up['d12'][z] = template_delta[12]
            
    print('raz1 = ', raz1)
    print('raz2 = ', raz2)
    mx = class_middle['Scalar_calc'].max()
    mn = class_middle['Scalar_calc'].min()
    #print('middle_max = ', class_middle['Scalar_calc'].max())
    #print('middle_min = ', class_middle['Scalar_calc'].min())
    print()
    #print(class_up)
    
    print('+++++++++++++++++++++++++++++++')
    class_up.drop(class_up.columns[0], axis=1, inplace=True)  
    class_down.drop(class_down.columns[0], axis=1, inplace=True)   
    class_middle.drop(class_middle.columns[0], axis=1, inplace=True)
    print(template_delta)
    print()
    print(weights_up)
    
class_up.to_csv('final/class_up.csv')
class_down.to_csv('final/class_down.csv')
class_middle.to_csv('final/class_middle.csv')

#if((sum_cut_safe == sum_truncate_current) and (sum_distance_safe = sum_distance_current)):
weights_up[0] += template_delta[0]
weights_up[1] += template_delta[1]
weights_up[2] += template_delta[2]
weights_up[3] += template_delta[3]
weights_up[4] += template_delta[4]
weights_up[5] += template_delta[5]
weights_up[6] += template_delta[6]
weights_up[7] += template_delta[7]
weights_up[8] += template_delta[8]
weights_up[9] += template_delta[9]
weights_up[10] += template_delta[10]
weights_up[11] += template_delta[11]
weights_up[12] += template_delta[12]
print()
print(weights_up)
    
    ##print(class_middle['Scalar_calc'][0])
    #print(distance_up)
    
    #print(sum_distance_current)
    #print(class_middle['Scalar_calc'][0])
    #print(class_middle['Scalar_calc'][len(class_middle['Scalar_calc'])-1])
    ##print(class_middle['Scalar_calc'][len(class_middle['Scalar_calc'])-1])
    #print(distance_down)
    #print()
    #print(weights_up)
    #print()
    #print(weights_up[0])
    #print(weights_up[1])
    #print(weights_up[2])
    #print(weights_up[3])
    #print(weights_up[4])
    #print(weights_up[5])
    #print(weights_up[6])
    #print(weights_up[7])
    #print(weights_up[8])
    #print(weights_up[9])
    #print(weights_up[10])
    #print(weights_up[11])
    #print(weights_up[12])
    #print()
    #print('__________________')
    
    
