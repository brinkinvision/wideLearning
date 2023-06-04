# wideLearning
![Логотип](/wideL.png)

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
