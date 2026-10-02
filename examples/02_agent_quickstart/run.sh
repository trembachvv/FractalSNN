#!/usr/bin/env bash
set -e

echo "=== Запуск Agent Quickstart Demo ==="
fpc -O3 -Mobjfpc -S2 -Si -Fu../../src agent_demo.pas
./agent_demo
rm -f *.o *.ppu
