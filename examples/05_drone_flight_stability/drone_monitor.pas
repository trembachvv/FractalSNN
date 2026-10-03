{ ============================================================================ }
{  Project:     FractalSNN Examples                                            }
{  File:        examples/05_drone_flight_stability/drone_monitor.pas           }
{  Description: UAV Flight Instability & Stall Early Detection (IMU 6-Axis)    }
{  Authors:     Trembach V.V.                       }
{  License:     Dual-License: GNU AGPLv3 / Commercial OEM                      }
{ ============================================================================ }

program drone_monitor;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  SysUtils, Math,
  fsnn.types, fsnn.matrix, fsnn.neuron, fsnn.datasets, fsnn.agent;

const
  TIME_STEPS = 15;
  IMU_CHANNELS = 6; // AccX, AccY, AccZ, GyroX, GyroY, GyroZ

function GenerateFlightDataset(Count: Integer): TDataset;
var
  s, t, c: Integer;
  isStall: Boolean;
begin
  Result.Samples := nil;
  SetLength(Result.Samples, Count);
  for s := 0 to Count - 1 do
  begin
    isStall := (s mod 2 = 1);
    Result.Samples[s].Label_ := Ord(isStall);
    SetLength(Result.Samples[s].Data, TIME_STEPS);

    for t := 0 to TIME_STEPS - 1 do
    begin
      SetLength(Result.Samples[s].Data[t], IMU_CHANNELS);
      for c := 0 to IMU_CHANNELS - 1 do
      begin
        if isStall then
          // Аномальные резкие угловые скорости и перегрузка
          Result.Samples[s].Data[t][c] := EnsureRange(0.5 + 0.45 * Sin(t * 1.2 + c) + (Random - 0.5) * 0.1, 0.0, 1.0)
        else
          // Стабильное висение / ламинарный полет
          Result.Samples[s].Data[t][c] := EnsureRange(0.2 + 0.05 * Sin(t * 0.2) + (Random - 0.5) * 0.05, 0.0, 1.0);
      end;
    end;
  end;
end;

var
  agent: TSpikeAgent;
  trainSet, valSet: TDataset;
  epoch, pred: Integer;
  valAcc: Double;
  emergencySample: array[0..5] of Double = (0.92, 0.88, 0.15, 0.95, 0.80, 0.75);
begin
  WriteLn('======================================================================');
  WriteLn('      FractalSNN UAV Edge: Раннее обнаружение срыва потока БПЛА       ');
  WriteLn('======================================================================');

  RandSeed := 1024;
  trainSet := GenerateFlightDataset(120);
  valSet   := GenerateFlightDataset(40);

  // 6 входов (IMU), 2 класса (Норма / Срыв), глубина фрактала 4
  agent := TSpikeAgent.Create(IMU_CHANNELS, 2, 4, 2);
  try
    WriteLn('Обучение f-LIF ядра на полетной телеметрии...');
    for epoch := 1 to 20 do
    begin
      agent.TrainEpoch(trainSet, valSet, 0.015, valAcc);
      if valAcc >= 0.95 then Break;
    end;
    WriteLn(Format('Обучение завершено. Точность детекции: %5.2f%%', [valAcc * 100]));

    WriteLn('Тестирование критического всплеска угловой скорости:');
    pred := agent.Predict(emergencySample);
    if pred = 1 then
      WriteLn('  [ТРЕВОГА] Зафиксирован срыв потока! Включить стабилизацию рулей.')
    else
      WriteLn('  [НОРМА] Полет в пределах коридора устойчивости.');
  finally
    agent.Free;
  end;
end.
