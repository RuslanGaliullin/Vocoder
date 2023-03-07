# Vocoder

Алгоритм предназначен для растягивания по времени цифрового аудио-сигнала без изменения питча.
Язык реализации: Python.
При помощи реализованного алгоритма можно растянуть в 2 раза и сжать в 2 раза (по длительности) прилагаемую аудиозапись.

#### Для запуска программы введите ``` ./run.sh [input.wav] [output.wav] [scale] ```
где 0 < scale < 1 для сжатия файла и scale >= 1 для растяжения
