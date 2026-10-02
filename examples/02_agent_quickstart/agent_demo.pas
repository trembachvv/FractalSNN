program agent_demo;

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  fsnn.types, fsnn.rng, fsnn.datasets, fsnn.agent;

var
  agent: TSpikeAgent;
  cfg: TExperimentConfig;
  trainData, valData: TDataset;
  epoch: Integer;
  valAcc: Double;
  stopped: Boolean;
  testInput: array[0..3] of Double;
  prediction: Integer;
begin
  InitRNG(42);
  cfg := DefaultConfig;

  WriteLn('=== Инициализация автономного агента FractalSNN ===');
  agent := TSpikeAgent.Create(4, 2, 4, 2); // 4 входа, 2 класса, глубина 4, поднейронов 2

  // Генерируем входной поток (например, Temporal XOR)
  trainData := GenerateTemporalXORDataset(200, 15, 4, 42, 0.05, 3);
  valData := GenerateTemporalXORDataset(50, 15, 4, 1042, 0.05, 3);

  WriteLn('Старт автономного цикла обучения...');
  epoch := 0;
  repeat
    Inc(epoch);
    stopped := agent.TrainEpoch(trainData, valData, 0.008, valAcc);
    WriteLn(Format('Эпоха %3d | Валидационная точность: %5.1f%%', [epoch, valAcc * 100]));
  until stopped or (epoch >= 50);

  WriteLn(Format('Обучение завершено по критерию останова на эпохе %d. Лучшая точность: %.1f%%',
    [epoch, agent.BestAccuracy * 100]));

  // Сохраняем готовую модель
  agent.SaveWeights('fractal_agent.bin');
  WriteLn('Веса сохранены в файл: fractal_agent.bin');

  // Боевой инференс на новых данных
  testInput[0] := 0.9; testInput[1] := 0.1; testInput[2] := 0.8; testInput[3] := 0.05;
  prediction := agent.Predict(testInput);
  WriteLn(Format('Предсказание для тестового вектора: Класс %d', [prediction]));

  agent.Free;
end.
