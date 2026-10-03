{ ============================================================================ }
{  Project:     FractalSNN Examples                                            }
{  File:        examples/01_vibration_anomaly/vibration_monitor.pas            }
{  Description: CNC Milling Spindle Predictive Maintenance Case                }
{  Authors:     Trembach V.V.                        }
{  License:     Dual-License: GNU AGPLv3 / Commercial OEM                      }
{ ============================================================================ }

program vibration_monitor;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  SysUtils, Math,
  fsnn.types, fsnn.matrix, fsnn.neuron, fsnn.datasets, fsnn.agent;

const
  TIME_WINDOW = 20;
  SENSOR_CHANS = 4;

function GenerateSpindleDataset(SampleCount: Integer): TDataset;
var
  s, t: Integer;
  isAnomaly: Boolean;
  freq, noise: Double;
begin
  Result.Samples := nil;
  SetLength(Result.Samples, SampleCount);
  for s := 0 to SampleCount - 1 do
  begin
    isAnomaly := (s mod 2 = 1);
    Result.Samples[s].Label_ := Ord(isAnomaly);
    SetLength(Result.Samples[s].Data, TIME_WINDOW);

    if isAnomaly then freq := 0.85 else freq := 0.20;

    for t := 0 to TIME_WINDOW - 1 do
    begin
      SetLength(Result.Samples[s].Data[t], SENSOR_CHANS);
      noise := (Random - 0.5) * 0.1;
      
      Result.Samples[s].Data[t][0] := EnsureRange(0.5 + 0.3 * Sin(t * freq) + noise, 0.0, 1.0);
      Result.Samples[s].Data[t][1] := EnsureRange(0.5 + 0.3 * Cos(t * freq) + noise, 0.0, 1.0);
      if isAnomaly then
        Result.Samples[s].Data[t][2] := EnsureRange(0.7 + noise, 0.0, 1.0)
      else
        Result.Samples[s].Data[t][2] := EnsureRange(0.2 + noise, 0.0, 1.0);
      Result.Samples[s].Data[t][3] := 0.45;
    end;
  end;
end;

var
  agent: TSpikeAgent;
  trainSet, testSet: TDataset;
  epoch, pred: Integer;
  valAcc: Double;
  damagedSignal: array[0..3] of Double = (0.85, 0.12, 0.90, 0.70);
begin
  WriteLn('======================================================================');
  WriteLn('   FractalSNN Industrial Case: CNC Spindle Predictive Maintenance     ');
  WriteLn('   Автор: Трембач В.В.                                                ');
  WriteLn('======================================================================');

  RandSeed := 12345;
  WriteLn('1. Генерация выборки вибрационных сигналов...');
  trainSet := GenerateSpindleDataset(160);
  testSet  := GenerateSpindleDataset(40);

  WriteLn('2. Инициализация спайкового ядра (4 канала, 4 уровня дендритов)...');
  agent := TSpikeAgent.Create(SENSOR_CHANS, 2, 4, 2);

  WriteLn('3. Старт обучения (BPTT / surrogate gradient)...');
  for epoch := 1 to 20 do
  begin
    agent.TrainEpoch(trainSet, testSet, 0.01, valAcc);
    if epoch mod 4 = 0 then
      WriteLn(Format('   [Эпоха %2d] Надежность распознавания: %5.2f%%', [epoch, valAcc * 100]));
    if valAcc >= 0.95 then
    begin
      WriteLn(Format('   >> Целевая точность достигнута на эпохе %d!', [epoch]));
      Break;
    end;
  end;

  WriteLn('');
  WriteLn('4. Тестирование на реальном всплеске осевой вибрации:');
  pred := agent.Predict(damagedSignal);
  if pred = 1 then
    WriteLn('   РЕЗУЛЬТАТ: [ТРЕВОГА] Зафиксирован развивающийся скол подшипника!')
  else
    WriteLn('   РЕЗУЛЬТАТ: [НОРМА] Вибрация в пределах допуска.');

  WriteLn('5. Сохранение весов модели: spindle_model.fsnn');
  agent.SaveCheckpoint('spindle_model.fsnn', epoch);

  agent.Free;
  WriteLn('Тестирование завершено успешно.');
end.