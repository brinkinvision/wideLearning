# wideLearning
Библиотека поиска оптимальной архитектуры нейронной сети.

# Установка
pip install widelearning

# Импорт библиотеки
Рекомендуется использовать следующую команду:
import widelearning as wdl 

# Описание функций
txt_kernel(file_path) - функция для установки собственного сверточного ядра. Параметр file_path является путем к текстовому файлу, в котором записано ядро. 
Ядро необходимо задавать в следующем виде (с запятыми в конце каждой строки):

1,2,3,  
4,5,6,  
7,8,9,  

Для установки ядра в слой необходимо выполнить следующую команду после компиляции модели:
model.layers[0].set_weights([weights])
где числом обозначается слой нейронной сети, в который происходит установка ядра.

vert(n) - генерация сверточного ядра вертикального типа.
hor(n) - генерация сверточного ядра горизонтального типа.
Параметр n представляет собой список, состоящий из значений, из которых будет сгенерировано ядро.
Например, если n = [-2, -1, 0, 1, 2], то ядра будут сгенерированы в следующем виде:

## wdl.vert([-2, -1, 0, 1, 2])
-2,-1,0,1,2,  
-2,-1,0,1,2,  
-2,-1,0,1,2,  
-2,-1,0,1,2,  
-2,-1,0,1,2,  

## wdl.hor([-2, -1, 0, 1, 2])
-2,-2,-2,-2,-2,  
-1,-1,-1,-1,-1,  
0,0,0,0,0,  
1,1,1,1,1,  
2,2,2,2,2,  

Список n также задает размерность генерируемого ядра (если список состоит из трех значений, то размерность ядра составит 3*3 и т.д.) 

diag(d) - генерация сверточного ядра диагонального типа. Параметр d задает размерность ядра.
В настоящий момент можно сгенерировать ядро любой размерности, начиная от 5*5, составленое из значений пятизначного диапазона [-2, -1, 0, 1, 2].  

## wdl.diag(7)
0,1,1,1,2,2,2,  
-1,0,1,1,1,2,2,  
-1,-1,0,1,1,1,2,  
-1,-1,-1,0,1,1,1,  
-2,-1,-1,-1,0,1,1,  
-2,-2,-1,-1,-1,0,1,  
-2,-2,-2,-1,-1,-1,0,  

2,2,2,1,1,1,0,  
2,2,1,1,1,0,-1,  
2,1,1,1,0,-1,-1,  
1,1,1,0,-1,-1,-1,  
1,1,0,-1,-1,-1,-2,  
1,0,-1,-1,-1,-2,-2,  
0,-1,-1,-1,-2,-2,-2,  

Генерируются два ядра: главная и побочная диагональ.
