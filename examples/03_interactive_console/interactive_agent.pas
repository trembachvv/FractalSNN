program interactive_agent;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  SysUtils, Classes, Math,
  fsnn.types, fsnn.rng, fsnn.datasets, fsnn.agent, fsnn.log;

type
  { Фоновый рабочий поток обучения }
  TTrainerThread = class(TThread)
  private
    FAgent: TSpikeAgent;
    FTrainData: TDataset;
    FValData: TDataset;
    FIsPaused: Boolean;
    FCurrentEpoch: Integer;
    FLastAcc: Double;
    procedure ShowAutoStopMsg;
  protected
    procedure Execute; override;
  public
    constructor Create(Agent: TSpikeAgent; const TrainData, ValData: TDataset);
    procedure PauseTraining;
    procedure ResumeTraining;
    function AskAgent(const Query: string): string;

    property IsPaused: Boolean read FIsPaused;
    property CurrentEpoch: Integer read FCurrentEpoch;
    property LastAcc: Double read FLastAcc;
  end;

{ TTrainerThread }

constructor TTrainerThread.Create(Agent: TSpikeAgent; const TrainData, ValData: TDataset);
begin
  inherited Create(True);
  FreeOnTerminate := False;
  FAgent := Agent;
  FTrainData := TrainData;
  FValData := ValData;
  FIsPaused := True;
  FCurrentEpoch := 0;
  FLastAcc := 0.0;
end;

procedure TTrainerThread.PauseTraining;
begin
  FIsPaused := True;
end;

procedure TTrainerThread.ResumeTraining;
begin
  FIsPaused := False;
  Start;
end;

procedure TTrainerThread.ShowAutoStopMsg;
begin
  WriteLn('');
  WriteLn(Format('[Auto-Stop] Обучение остановлено на эпохе %d. Лучшая точность: %.2f%%',
    [FCurrentEpoch, FLastAcc * 100]));
  Write('> ');
  Flush(Output);
end;

function TTrainerThread.AskAgent(const Query: string): string;
var
  inVec: array[0..7] of Double;
  i, predClass: Integer;
  hVal: LongWord;
begin
  hVal := 2166136261;
  for i := 1 to Length(Query) do
    hVal := (hVal xor Ord(Query[i])) * 16777619;

  for i := 0 to 7 do
    inVec[i] := ((hVal shr ((i mod 4) * 8)) and $FF) / 255.0;

  predClass := FAgent.Predict(inVec);

  case predClass of
    0: Result := Format('[Класс 0] Архитектура стандартная/алгоритмическая (Лучшая Acc=%.1f%%).',
         [FAgent.BestAccuracy * 100]);
    1: Result := Format('[Класс 1] Обнаружен асинхронный синтаксис / специфический веб-паттерн (Лучшая Acc=%.1f%%).',
         [FAgent.BestAccuracy * 100]);
  else
    Result := 'Паттерн не распознан.';
  end;
end;

procedure TTrainerThread.Execute;
var
  stopped: Boolean;
  valAcc: Double;
begin
  while not Terminated do
  begin
    if not FIsPaused then
    begin
      Inc(FCurrentEpoch);
      stopped := FAgent.TrainEpoch(FTrainData, FValData, 0.005, valAcc);
      FLastAcc := valAcc;

      if stopped then
      begin
        FIsPaused := True;
        ShowAutoStopMsg;
      end;

      Sleep(10);
    end
    else
      Sleep(100);
  end;
end;

procedure RunRepl;
var
  agent: TSpikeAgent;
  trainer: TTrainerThread;
  fullSet, trainSet, valSet: TDataset;
  cmd, arg: string;
  resEpoch, splitIdx, i: Integer;
begin
  InitRNG(42);
  WriteLn('======================================================================');
  WriteLn('    Интерактивная консоль управления обучением FractalSNN v1.4        ');
  WriteLn('======================================================================');
  WriteLn('Команды:');
  WriteLn('  /start          - Запустить фоновое обучение');
  WriteLn('  /pause          - Поставить обучение на паузу');
  WriteLn('  /resume         - Возобновить обучение');
  WriteLn('  /status         - Текущая эпоха, точность и состояние сети');
  WriteLn('  /save <файл>    - Сохранить чекпоинт весов');
  WriteLn('  /load <файл>    - Загрузить чекпоинт весов');
  WriteLn('  /ask <текст>    - Анализ фрагмента кода спайковой моделью');
  WriteLn('  /exit           - Завершить программу');
  WriteLn('----------------------------------------------------------------------');

  if FileExists('github_code.csv') then
  begin
    fullSet := LoadDatasetFromCSV('github_code.csv', 20, 8);
    if Length(fullSet.Samples) >= 10 then
    begin
      WriteLn(Format('Загружен github_code.csv (%d сэмплов).', [Length(fullSet.Samples)]));
      splitIdx := Max(1, Trunc(Length(fullSet.Samples) * 0.8));
      SetLength(trainSet.Samples, splitIdx);
      for i := 0 to splitIdx - 1 do
        trainSet.Samples[i] := fullSet.Samples[i];

      SetLength(valSet.Samples, Length(fullSet.Samples) - splitIdx);
      for i := splitIdx to High(fullSet.Samples) do
        valSet.Samples[i - splitIdx] := fullSet.Samples[i];

      agent := TSpikeAgent.Create(8, 2, 4, 2);
    end
    else
    begin
      WriteLn('В github_code.csv недостаточно строк. Используется базовый датасет TemporalXOR.');
      trainSet := GenerateTemporalXORDataset(300, 15, 8, 42, 0.05, 3);
      valSet   := GenerateTemporalXORDataset(80, 15, 8, 1042, 0.05, 3);
      agent := TSpikeAgent.Create(8, 2, 4, 2);
    end;
  end
  else
  begin
    WriteLn('github_code.csv не найден. Инициализация базового набора (TemporalXOR)...');
    trainSet := GenerateTemporalXORDataset(300, 15, 8, 42, 0.05, 3);
    valSet   := GenerateTemporalXORDataset(80, 15, 8, 1042, 0.05, 3);
    agent := TSpikeAgent.Create(8, 2, 4, 2);
  end;

  trainer := TTrainerThread.Create(agent, trainSet, valSet);

  while True do
  begin
    Write('> ');
    ReadLn(cmd);
    cmd := Trim(cmd);
    if cmd = '' then Continue;

    if cmd = '/exit' then
    begin
      WriteLn('Остановка потока и завершение...');
      trainer.Terminate;
      trainer.WaitFor;
      Break;
    end
    else if cmd = '/start' then
    begin
      trainer.ResumeTraining;
      WriteLn('Обучение запущено в фоновом режиме.');
    end
    else if cmd = '/pause' then
    begin
      trainer.PauseTraining;
      WriteLn(Format('Обучение приостановлено на эпохе %d.', [trainer.CurrentEpoch]));
    end
    else if cmd = '/resume' then
    begin
      trainer.ResumeTraining;
      WriteLn('Обучение возобновлено.');
    end
    else if cmd = '/status' then
    begin
      WriteLn(Format('Статус: [Пауза=%s] | Эпоха: %d | Лучшая точность: %.2f%% | Актуальная: %.2f%%',
        [BoolToStr(trainer.IsPaused, 'Да', 'Нет'), trainer.CurrentEpoch,
         agent.BestAccuracy * 100, trainer.LastAcc * 100]));
    end
    else if Copy(cmd, 1, 6) = '/save ' then
    begin
      arg := Trim(Copy(cmd, 7, Length(cmd)));
      agent.SaveCheckpoint(arg, trainer.CurrentEpoch);
    end
    else if Copy(cmd, 1, 6) = '/load ' then
    begin
      arg := Trim(Copy(cmd, 7, Length(cmd)));
      trainer.PauseTraining;
      if agent.LoadCheckpoint(arg, resEpoch) then
        WriteLn(Format('Модель загружена с эпохи %d.', [resEpoch]))
      else
        WriteLn('Не удалось прочитать файл чекпоинта.');
    end
    else if Copy(cmd, 1, 5) = '/ask ' then
    begin
      arg := Trim(Copy(cmd, 6, Length(cmd)));
      WriteLn('Ответ FractalSNN: ' + trainer.AskAgent(arg));
    end
    else
    begin
      WriteLn('Ответ FractalSNN: ' + trainer.AskAgent(cmd));
    end;
  end;

  trainer.Free;
  agent.Free;
  WriteLn('Сеанс завершен.');
end;

begin
  RunRepl;
end.