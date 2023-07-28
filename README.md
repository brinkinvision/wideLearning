# wideLearning
![Логотип](/wideL.png)

[**`Инструкция`**](https://github.com/brinkinvision/wideLearning/tree/main/Туториал) | [**`GitFlic`**](https://gitflic.ru/project/brinkinvision/wide-learning) | [**`PyPI`**](https://pypi.org/project/widelearning/) | [**`Пример новых вариантов использования`**](https://github.com/brinkinvision/wideLearning/tree/main/Обучение_на_ошибках)

# Публикации
1. Гиниятуллин В.М., Ермолаев Е.В., Хлыбов А.В. ФОРМАЛИЗАЦИЯ ПРОЦЕДУРЫ ПОДБОРА СВЕРТОЧНЫХ ЯДЕР. Вестник кибернетики. 2022;(2 (46)):66-74. https://doi.org/10.34822/1999-7604-2022-2-66-74
```
https://www.vestcyber.ru/jour/article/view/437
```
2. Гиниятуллин В.М., Хлыбов А.В., Федоров М.А., Асадуллин Т.А., Крутин А.С., Осипов И.А., Зарипов Д.М. СРАВНЕНИЕ ТИПОВ ЯДЕР В СВЕРТОЧНЫХ СЛОЯХ НЕЙРОННЫХ СЕТЕЙ. Вестник кибернетики. 2022;(3 (47)):84-98. https://doi.org/10.34822/1999-7604-2022-3-84-98
```
https://www.vestcyber.ru/jour/article/view/461
```

# Минимальные требования к аппаратной части:
* Операционная система: Windows 7 или выше (64-bit), Linux (64-bit)
* Оперативная память: не менее 2 GB
* Процессор: двухъядерный с частотой 1.6 GHz или выше
* Свободное место на жестком диске: не менее 2 GB

# Направления прикладного использования:
1. Компьютерное зрение. Упрощение архитектур сверточных нейронных сетей для задач классификации и распознавания изображений, сегментации и детектирования объектов.
2. Медицинская диагностика. Оптимизация нейронных сетей, используемых при анализе изображений и сигналов, полученных от медицинских приборов. Может использоваться для обнаружения и классификации заболеваний.
3. Рекомендательные системы. Может использоваться для классификации на основе некоторых признаков контента и выдачи персонализированных рекомендаций.
4. Развертывание моделей нейронных сетей на устройствах малой вычислительной мощности, например, мобильные гаджеты (смартфоны/планшеты), встраиваемые устройства (промышленные контроллеры/смарт-устройства) и т.п.

# Установка
```
pip install widelearning
```

# Импорт библиотеки
```
import widelearning as wdl 
```

# Описание функций

### txt_kernel(file_path)	
    
    Функция преобразования сгенерированного ядра для вставки в структуру TensorFlow 
    (для черно-белых, одноканальных изображений)

    Параметры
    ----------
    file_path : str
        путь к текстовому файлу, содержащему сверточное ядро, записанное в следующем виде 
        (с запятыми в конце каждой строки)
        1,2,3,
        4,5,6,
        7,8,9,

    Пример использования:
    wdl.txt_kernel('horizontal.txt')

### txt_kernel_rgb(k1, k2, k3) 
    
    Функция преобразования сгенерированного ядра для вставки в структуру TensorFlow 
    (для цветных, трехканальных изображений)
    
    Параметры
    ----------
    k1 : str
        путь к текстовому файлу, содержащему сверточное ядро первого цветового канала R, записанное в следующем виде 
        (с запятыми в конце каждой строки)
        1,2,3,
        4,5,6,
        7,8,9,
    k2 : str
        путь к текстовому файлу, содержащему сверточное ядро второго цветового канала G
    k3 : str
        путь к текстовому файлу, содержащему сверточное ядро третьего цветового канала B

    Пример использования:
    wdl.txt_kernel_rgb('k1.txt', 'k2.txt', 'k3.txt')

### add_kernel(path, init_kernel)
    
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
    
### add_kernel_rgb(k1, k2, k3, w)
    
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

### horizontal(dim, values)
    
    Функция для генерации линейного сверточного ядра горизонтального типа

    Параметры
    ----------
    dim : int
        размерность генерируемой матрицы 
    values : list
        список значений, которые будут использоваться для заполнения матрицы.

    Пример использования:
    wdl.horizontal(32, [-7,-5,-3,-1,1,3,5,7])
	
### vertical(dim, values)
    
    Функция для генерации линейного сверточного ядра вертикального типа

    Параметры
    ----------
    dim : int
        размерность генерируемой матрицы 
    values : list
        список значений, которые будут использоваться для заполнения матрицы.

    Пример использования:
    wdl.vertical(32, [-7,-5,-3,-1,1,3,5,7])

### diagonal(up, down, d, dimension)
    
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
	
### show_kernel(N, name)
    
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

### select_top(path, label)
    
    Функция позволяет получить грубое первоначальное приближение вектора весов

    Параметры
    ----------
    path : str
        путь к целочисленной обучающей выборке в формате csv
    label : str
        название столбца с метками классов

    Пример использования:
    wdl.select_top('KAHRAMAN_train.csv', 'UNS')

### select_top_binary(path, label)
    
    Функция позволяет получить грубое первоначальное приближение вектора весов для бинарной обучающей выборки

    Параметры
    ----------
    path : str
        путь к целочисленной бинарной обучающей выборке в формате csv
    label : str
        название столбца с метками классов

    Пример использования:
    wdl.select_top_binary('train_6.csv', 'UNS')

### grad_boost(path, t, oth, l, s, from_initial, weights_main)
    
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

### grad_boost_save_files(path, t, oth, l, s, from_initial, weights_main)
    
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

### grad(path, t, oth, l, s, weights)
    
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
	
### grad_binary(path, t, oth, l, s, weights_main)
    
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

### data_int(path, decimal, name)
    
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

### scale_weights(label, data_folder, weights_folder)
    
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

### generate_fa(label, data_folder, weights_folder)
    
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

### check_test(train, weights, test, label)
    
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

### getNameColumn(nameFileTrain)
    
    Вспомогательная функция для определения названий всех столбцов, кроме столбца с метками классов

    Параметры
    ----------
    nameFileTrain : str
        путь к обучающей выборке 

### getCountNeuronSizeVector(nameFileTrain)
    
    Вспомогательная функция для определения необходимого количества нейронов в слое

    Параметры
    ----------
    nameFileTrain : str
        путь к обучающей выборке 

### initializingDictionaryKeys()
    
    Вспомогательная функция для инициализации ключей для правила логического вывода

### initializationDictionaryValues(nameFileTrain)
    
    Вспомогательная функция для создания словаря логического вывода

    Параметры
    ----------
    nameFileTrain : str
        путь к обучающей выборке 

### count_conv_operations(height, width, color_channels, kernel_size, stride, num_kernels)
    
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

# Шаблоны вызова функций сверточного слоя
```
wdl.txt_kernel('путь_к_сгенерированному_ядру.txt')
```
```
wdl.txt_kernel_rgb('путь_к_сгенерированному_ядру_ПЕРВОГО_цветового_канала.txt', 'путь_к_сгенерированному_ядру_ВТОРОГО_цветового_канала.txt', 'путь_к_сгенерированному_ядру_ТРЕТЬЕГО_цветового_канала.txt')
```
```
wdl.add_kernel('путь_к_добавляемому_ядру.txt', переменная_уже_установленных_ядер)
```
```
wdl.add_kernel_rgb('путь_к_добавляемому_ядру_ПЕРВОГО_цветового_канала.txt', 'путь_к_добавляемому_ядру_ВТОРОГО_цветового_канала.txt', 'путь_к_добавляемому_ядру_ТРЕТЬЕГО_цветового_канала.txt', переменная_уже_установленных_ядер)
```
```
wdl.horizontal(размерность_ядра, [список_значений_ядра])
```
```
wdl.vertical(размерность_ядра, [список_значений_ядра])
```
```
wdl.diagonal([список_значений_выше_диагонали], [список_значений_ниже_диагонали], значение_диагонали, размерность_ядра)
```
```
wdl.show_kernel(номер_сверточного_слоя, объект_модели)
```
```
wdl.count_conv_operations(высота_изображения, ширина_изображения, количество_цветовых_каналов, размерность_квадратного_ядра, шаг_ядра, количество_ядер)
```

# Шаблоны вызова функций полносвязного слоя
```
wdl.data_int('путь_к_обучающей_выборке.csv', 'путь_к_файлу_со_значениями_кол-ва_знаков_после_запятой.txt', 'название_выходных_файлов')
```
```
wdl.select_top('путь_к_целочисленной_обучающей_выборке.csv', 'название_столбца_с_метками_классов')
```
```
wdl.select_top_binary('путь_к_целочисленной_обучающей_выборке.csv', 'название_столбца_с_метками_классов')
```
```
wdl.grad('путь_к_целочисленной_обучающей_выборке.csv', [целевой_класс], [список_остальных_классов], 'название_столбца_с_метками_классов', шаг, [вектор_весов])
```
```
wdl.grad_binary('путь_к_целочисленной_обучающей_выборке.csv', [целевой_класс], [список_остальных_классов], 'название_столбца_с_метками_классов', шаг, [вектор_весов])
```
```
wdl.scale_weights('название_столбца_с_метками_классов', 'путь_к_папке_с_обучающими_выборками', 'путь_к_папке_с_весами')
```
```
wdl.generate_fa('название_столбца_с_метками_классов', 'путь_к_папке_result', 'путь_к_папке_scale')
```
```
wdl.check_test('путь_к_полной_обучающей_выборке.csv', 'путь_к_файлу_со_всеми_весами.txt', 'путь_к_тестовой_выборке.csv', 'название_столбца_с_метками_классов')
```
