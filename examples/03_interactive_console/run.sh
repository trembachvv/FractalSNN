#!/usr/bin/env bash
set -e

echo "=== Запуск интерактивной консоли управления FractalSNN ==="
fpc -O3 -Mobjfpc -S2 -Si -Fu../../src interactive_agent.pas
./interactive_agent
rm -f *.o *.ppu
