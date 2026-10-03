#!/usr/bin/env bash
set -e

echo "=== Сборка и запуск кейса: Вибродиагностика шпинделя ЧПУ ==="

fpc -O3 -Mobjfpc -S2 -Si -Fu../../src drone_monitor.pas

START_TIME=$(date +%s%N)
./drone_monitor
END_TIME=$(date +%s%N)

# Время инференса и обучения в миллисекундах
DIFF_MS=$(( (END_TIME - START_TIME) / 1000000 ))

rm -f *.o *.ppu

echo ""
echo "=== Время полного цикла (обучение + инференс): ${DIFF_MS} мс ==="
