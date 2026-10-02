# Справочник API библиотеки FractalSNN

## Модули ядра (`src/`)

| Модуль              | Назначение |
| :---                | :--- |
| `fsnn.types.pas`    | Перечисления конфигураций, типы `TExperimentConfig`, `TRunResult`.                           |
| `fsnn.matrix.pas`   | Линейная алгебра, непрерывные плотные матрицы `TMatrix`, шаг оптимизатора `AdamStep`.        |
| `fsnn.neuron.pas`   | Реализация многоуровневого дендритного нейрона `TFractalNeuron`.                             |
| `fsnn.network.pas`  | Связка сети `TSNNNetwork` (скрытый + выходной слои), прямой проход и дифференцирование BPTT. |
| `fsnn.datasets.pas` | Генераторы синтетических бенчмарков (`TemporalXOR`, `DelayedMatch`) и парсер CSV.            |
| `fsnn.agent.pas`    | Класс `TSpikeAgent` для автономного обучения, квантования и сохранения чекпоинтов `.fsnn`.   |
| `fsnn.threads.pas`  | Пул потоков ОС (`cthreads`) для параллельного прогона сидов `RunSeedsParallel`.              |
