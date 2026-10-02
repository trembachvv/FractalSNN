#!/usr/bin/env bash
set -e

echo "Очистка репозитория FractalSNN от временных файлов..."

# 1. Артефакты компилятора FPC
find . -type f \( -name "*.o" -o -name "*.ppu" -o -name "*.a" -o -name "*.bak" -o -name "link.res" \) -delete

# 2. Логи и временные файлы тестов
find . -type f \( -name "*.log" -o -name "*_temp.csv" -o -name "ci_test_run.*" \) -delete

# 3. Исполняемые файлы
rm -f tests/test_suite
rm -f benchmarks/run_benchmarks
rm -f examples/01_vibration_anomaly/vibration_monitor
rm -f examples/02_agent_quickstart/agent_demo
rm -f examples/03_interactive_console/interactive_agent
rm -f examples/04_github_harvester/harvest_github

echo "Репозиторий полностью очищен!"
