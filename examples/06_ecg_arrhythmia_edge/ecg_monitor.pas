program ecg_monitor;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  SysUtils, Math, DateUtils,
  fsnn.types, fsnn.matrix, fsnn.neuron, fsnn.datasets, fsnn.agent;

const
  ECG_WINDOW = 20;
  INPUT_DIM  = 2; // [Амплитуда, Производная dV]

function GenerateECGDataset(Count: Integer; BaseSeed: Integer): TDataset;
var
  s, t: Integer;
  isArrhythmia: Boolean;
  peakPos: Integer;
  v, prevV, dv: Double;
begin
  RandSeed := BaseSeed;
  Result.Samples := nil;
  SetLength(Result.Samples, Count);
  for s := 0 to Count - 1 do
  begin
    isArrhythmia := (s mod 2 = 1);
    Result.Samples[s].Label_ := Ord(isArrhythmia);
    SetLength(Result.Samples[s].Data, ECG_WINDOW);

    // При аритмии зубец R идет раньше (экстрасистола)
    if isArrhythmia then peakPos := 5 else peakPos := 13;

    prevV := 0.0;
    for t := 0 to ECG_WINDOW - 1 do
    begin
      SetLength(Result.Samples[s].Data[t], INPUT_DIM);

      if t = peakPos then
        v := 0.95
      else if (t = peakPos - 1) or (t = peakPos + 1) then
        v := 0.45
      else
        v := EnsureRange(Random * 0.05, 0.0, 1.0);

      dv := EnsureRange((v - prevV + 1.0) * 0.5, 0.0, 1.0); // нормализованная разность
      prevV := v;

      Result.Samples[s].Data[t][0] := v;
      Result.Samples[s].Data[t][1] := dv;
    end;
  end;
end;

var
  agent: TSpikeAgent;
  trainSet, valSet: TDataset;
  epoch, pred, t: Integer;
  valAcc, bestAcc: Double;
  flatInput: array[0..(ECG_WINDOW * INPUT_DIM) - 1] of Double;
  v, prevV, dv: Double;
  t0: TDateTime;
begin
  WriteLn('======================================================================');
  WriteLn('      FractalSNN MedTech: Носимый спайковый анализатор ЭКГ           ');
  WriteLn('======================================================================');

  t0 := Now;
  trainSet := GenerateECGDataset(160, 42);
  valSet   := GenerateECGDataset(40, 1024);

  // 2 входа (v, dv), 2 класса (норма/аритмия), глубина D=4, масштаб S=2
  agent := TSpikeAgent.Create(INPUT_DIM, 2, 4, 2);
  try
    WriteLn('Обучение f-LIF ядра на фазовых кардио-паттернах (BPTT)...');
    bestAcc := 0.0;
    for epoch := 1 to 50 do
    begin
      agent.TrainEpoch(trainSet, valSet, 0.05, valAcc);
      if valAcc > bestAcc then bestAcc := valAcc;

      if (epoch mod 5 = 0) or (valAcc >= 0.95) then
        WriteLn(Format('  Эпоха %2d: точность валидации = %5.2f%%', [epoch, valAcc * 100]));

      if valAcc >= 0.95 then
      begin
        WriteLn('  [Конвергенция достигнута: точность >= 95%]');
        Break;
      end;
    end;
    WriteLn(Format('Обучение завершено. Итоговая точность: %5.2f%%', [bestAcc * 100]));

    // Тестовый сигнал с ранней желудочковой экстрасистолой (peakPos = 5)
    prevV := 0.0;
    for t := 0 to ECG_WINDOW - 1 do
    begin
      if t = 5 then v := 0.95
      else if (t = 4) or (t = 6) then v := 0.45
      else v := 0.02;

      dv := EnsureRange((v - prevV + 1.0) * 0.5, 0.0, 1.0);
      prevV := v;

      flatInput[t * INPUT_DIM + 0] := v;
      flatInput[t * INPUT_DIM + 1] := dv;
    end;

    pred := agent.Predict(flatInput);
    if pred = 1 then
      WriteLn('  [ТРЕВОГА] Зафиксирована желудочковая экстрасистолия!')
    else
      WriteLn('  [НОРМА] Синусовый ритм в норме.');

    WriteLn(Format(#10'=== Время полного цикла: %d мс ===', [MilliSecondsBetween(Now, t0)]));
  finally
    agent.Free;
  end;
end.