
import numpy as np

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
    
def vert(n):
    for i in range(len(n)):
        print(*[x for x in n],'',sep=',')

def hor(n):
    hr = []
    for i in n:
        hr.append(len(n)*[i])
	
    for k1 in range(len(hr)):
        print()
        for k2 in range(len(hr[k1])):
            print(hr[k1][k2], end=',')
            
def diag(d):
    n = [0, -2, -1, 1, 2]

    if(d % 2 == 0):
        m1 = d//2
        m2 = (d/2) - 1
    else:
        m1 = d // 2
        m2 = d // 2

    L = {}

    c = 1
    for i1 in range(int(m2)):
        L[c] = [n[3]]*(d-c) 
        c += 1
    for i2 in range(int(m1)):
        L[c] = [n[4]]*(d-c)
        c += 1
    
    c = -1
    for j1 in range(int(m2)):
        L[c] = [n[2]]*(d+c) 
        c -= 1
    for j2 in range(int(m1)):
        L[c] = [n[1]]*(d+c)
        c -= 1

    diag1 = 0
    for u in L:
        diag1 = diag1 + np.diag(L[u],u)
        z1 = np.diag(np.full(d,n[0]))+diag1
        
    z2 = np.fliplr(z1)

    for h in range(len(z1)):
        print()
        for g in range(len(z1[h])):
            print(z1[h][g], end=',')
            
    print()
    
    for hh in range(len(z2)):
        print()
        for gg in range(len(z2[hh])):
            print(z2[hh][gg], end=',')

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
    
def horizontal(dim, values):
    matrix = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(values[i % len(values)])
        matrix.append(row)
		
	with open('horizontal.txt', 'w') as f:
		for row in z:
			f.write(','.join(str(x) for x in row) + ',\n')
	
    return matrix

def vertical(dim, values):
    matrix = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(values[j % len(values)])
        matrix.append(row)
		
	with open('vertical.txt', 'w') as f:
		for row in z:
			f.write(','.join(str(x) for x in row) + ',\n')
	
    return matrix

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
		for i in result:
			for j in i:
				file.write(str(j) + ',')
			file.write('\n')
	
	matrix2 = np.fliplr(result)
	
	with open('diagonal2.txt', 'w') as file:
		for i in matrix2:
			for j in i:
				file.write(str(j) + ',')
			file.write('\n')
	
    return matrix

def show_kernel(n):
	ff = model.layers[0].get_weights()

	for nn in range(len(ff[0][0][0][0])):
	print('\n' + f'Номер сверточного ядра: {nn}')
	for kk in range(len(ff[0][0][0])):
		for i in range(len(ff[0])):
			for j in range(len(ff[0][i])):
        
			print(ff[0][i][j][kk][nn], end=',')
		print()
		print('\n')

