#!/usr/bin/env bash
set -e

echo "=== Запуск GitHub Harvester (Free Pascal HTTP Client) ==="
fpc -O2 -Mobjfpc -S2 -Fu../../src harvest_github.pas
./harvest_github
rm -f *.o *.ppu
