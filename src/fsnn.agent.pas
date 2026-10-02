unit fsnn.agent;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Classes, Math,
  fsnn.types, fsnn.rng, fsnn.matrix, fsnn.datasets,
  fsnn.neuron, fsnn.network, fsnn.train, fsnn.log;

type
  { Заголовок бинарного файла чекпоинта }
  TCheckpointHeader = packed record
    Magic: array[0..3] of AnsiChar; // 'FSNN'
    Epoch: Integer;
    BestAccuracy: Double;
    PatienceCounter: Integer;
  end;

  { Кодировщик вещественных сигналов в спайковые последовательности }
  TSpikeEncoder = class
  public
    class function EncodeRate(const RealValues: array of Double; TimeSteps: Integer): TSample;
  end;

  { Автономный обучающийся агент с контролем сходимости }
  TSpikeAgent = class
  private
    FConfig: TExperimentConfig;
    FNet: TSNNNetwork;
    FBestAccuracy: Double;
    FPatienceCounter: Integer;
    FMaxPatience: Integer;
  public
    constructor Create(InFeatures, OutClasses: Integer; Depth: Integer = 4; Sub: Integer = 2);
    destructor Destroy; override;

    function TrainEpoch(const TrainSet, ValSet: TDataset; LR: Double; out ValAcc: Double): Boolean;
    function Predict(const RealValues: array of Double): Integer;
    
    procedure SaveWeights(const FileName: string);
    procedure LoadWeights(const FileName: string);
    procedure SaveCheckpoint(const FileName: string; CurrentEpoch: Integer);
    function LoadCheckpoint(const FileName: string; out ResumedEpoch: Integer): Boolean;

    property Network: TSNNNetwork read FNet;
    property BestAccuracy: Double read FBestAccuracy;
    property PatienceCounter: Integer read FPatienceCounter;
  end;

{ Функция загрузки CSV в TDataset }
function LoadDatasetFromCSV(const FileName: string; TimeSteps, Features: Integer): TDataset;

implementation

{ TSpikeEncoder }

class function TSpikeEncoder.EncodeRate(const RealValues: array of Double; TimeSteps: Integer): TSample;
var
  t, i, n: Integer;
  prob: Double;
begin
  FillChar(Result, SizeOf(Result), 0);
  n := Length(RealValues);
  SetLength(Result.Data, TimeSteps);
  Result.Label_ := 0;

  for t := 0 to TimeSteps - 1 do
  begin
    SetLength(Result.Data[t], n);
    for i := 0 to n - 1 do
    begin
      prob := EnsureRange(RealValues[i], 0.0, 1.0);
      if RngUniform(ThreadRng) < prob then
        Result.Data[t][i] := 1.0
      else
        Result.Data[t][i] := 0.0;
    end;
  end;
end;

{ TSpikeAgent }

constructor TSpikeAgent.Create(InFeatures, OutClasses: Integer; Depth: Integer; Sub: Integer);
begin
  inherited Create;
  FConfig := DefaultConfig;
  FConfig.InputSize := InFeatures;
  FConfig.OutputSize := OutClasses;
  FConfig.FractalDepth := Depth;
  FConfig.SubNeuronsPerLvl := Sub;
  FConfig.TimeConstants := tcMultiScale;
  FConfig.BetaMin := 0.85;
  FConfig.BetaMax := 0.995;

  FNet := TSNNNetwork.Create(FConfig);
  FBestAccuracy := 0.0;
  FPatienceCounter := 0;
  FMaxPatience := 12;
end;

destructor TSpikeAgent.Destroy;
begin
  FNet.Free;
  inherited Destroy;
end;

function TSpikeAgent.TrainEpoch(const TrainSet, ValSet: TDataset; LR: Double; out ValAcc: Double): Boolean;
var
  s: Integer;
begin
  for s := 0 to High(TrainSet.Samples) do
    FNet.TrainSample(TrainSet.Samples[s], LR);

  ValAcc := FNet.Evaluate(ValSet);

  if ValAcc > FBestAccuracy + 0.005 then
  begin
    FBestAccuracy := ValAcc;
    FPatienceCounter := 0;
    Result := False;
  end
  else
  begin
    Inc(FPatienceCounter);
    Result := (FPatienceCounter >= FMaxPatience);
  end;
end;

function TSpikeAgent.Predict(const RealValues: array of Double): Integer;
var
  sample: TSample;
  logits: TDoubleArray;
  i, bestIdx: Integer;
begin
  sample := TSpikeEncoder.EncodeRate(RealValues, FConfig.TimeSteps);
  logits := FNet.ForwardSample(sample);

  bestIdx := 0;
  for i := 1 to High(logits) do
    if logits[i] > logits[bestIdx] then
      bestIdx := i;

  Result := bestIdx;
end;

procedure TSpikeAgent.SaveWeights(const FileName: string);
var
  fs: TFileStream;
  l, i, count: Integer;
  val: Double;
begin
  fs := TFileStream.Create(FileName, fmCreate);
  try
    for l := 0 to FNet.HidFractal.DepthVal - 1 do
    begin
      count := FNet.HidFractal.WeightRows(l) * FNet.HidFractal.WeightCols(l);
      for i := 0 to count - 1 do
      begin
        val := FNet.HidFractal.WeightAt(l, i);
        fs.WriteBuffer(val, SizeOf(Double));
      end;
    end;
  finally
    fs.Free;
  end;
end;

procedure TSpikeAgent.LoadWeights(const FileName: string);
var
  fs: TFileStream;
  l, i, count: Integer;
  val: Double;
begin
  if not FileExists(FileName) then Exit;
  fs := TFileStream.Create(FileName, fmOpenRead);
  try
    for l := 0 to FNet.HidFractal.DepthVal - 1 do
    begin
      count := FNet.HidFractal.WeightRows(l) * FNet.HidFractal.WeightCols(l);
      for i := 0 to count - 1 do
      begin
        fs.ReadBuffer(val, SizeOf(Double));
        FNet.HidFractal.SetWeightAt(l, i, val);
      end;
    end;
  finally
    fs.Free;
  end;
end;

procedure TSpikeAgent.SaveCheckpoint(const FileName: string; CurrentEpoch: Integer);
var
  fs: TFileStream;
  hdr: TCheckpointHeader;
  l, i, count: Integer;
  val: Double;
begin
  fs := TFileStream.Create(FileName, fmCreate);
  try
    hdr.Magic := 'FSNN';
    hdr.Epoch := CurrentEpoch;
    hdr.BestAccuracy := FBestAccuracy;
    hdr.PatienceCounter := FPatienceCounter;
    fs.WriteBuffer(hdr, SizeOf(hdr));

    for l := 0 to FNet.HidFractal.DepthVal - 1 do
    begin
      count := FNet.HidFractal.WeightRows(l) * FNet.HidFractal.WeightCols(l);
      for i := 0 to count - 1 do
      begin
        val := FNet.HidFractal.WeightAt(l, i);
        fs.WriteBuffer(val, SizeOf(Double));
      end;
    end;
  finally
    fs.Free;
  end;
  LogFmt('Чекпоинт сохранен: %s (Эпоха %d, Acc: %.2f%%)', [FileName, CurrentEpoch, FBestAccuracy * 100]);
end;

function TSpikeAgent.LoadCheckpoint(const FileName: string; out ResumedEpoch: Integer): Boolean;
var
  fs: TFileStream;
  hdr: TCheckpointHeader;
  l, i, count: Integer;
  val: Double;
begin
  Result := False;
  if not FileExists(FileName) then Exit;
  fs := TFileStream.Create(FileName, fmOpenRead);
  try
    fs.ReadBuffer(hdr, SizeOf(hdr));
    if hdr.Magic <> 'FSNN' then Exit;

    ResumedEpoch := hdr.Epoch;
    FBestAccuracy := hdr.BestAccuracy;
    FPatienceCounter := hdr.PatienceCounter;

    for l := 0 to FNet.HidFractal.DepthVal - 1 do
    begin
      count := FNet.HidFractal.WeightRows(l) * FNet.HidFractal.WeightCols(l);
      for i := 0 to count - 1 do
      begin
        fs.ReadBuffer(val, SizeOf(Double));
        FNet.HidFractal.SetWeightAt(l, i, val);
      end;
    end;
    Result := True;
    LogFmt('Чекпоинт загружен: %s (Эпоха %d, Лучшая Acc: %.2f%%)', [FileName, ResumedEpoch, FBestAccuracy * 100]);
  finally
    fs.Free;
  end;
end;

function LoadDatasetFromCSV(const FileName: string; TimeSteps, Features: Integer): TDataset;
var
  lines, tokens: TStringList;
  i, t, f, valIdx: Integer;
  fs: TFormatSettings;
begin
  FillChar(Result, SizeOf(Result), 0);
  fs.DecimalSeparator := '.';
  lines := TStringList.Create;
  tokens := TStringList.Create;
  try
    lines.LoadFromFile(FileName);
    SetLength(Result.Samples, lines.Count);

    for i := 0 to lines.Count - 1 do
    begin
      tokens.Clear;
      tokens.Delimiter := ',';
      tokens.StrictDelimiter := True;
      tokens.DelimitedText := lines[i];

      if tokens.Count < 1 + TimeSteps * Features then Continue;

      Result.Samples[i].Label_ := StrToIntDef(tokens[0], 0);
      SetLength(Result.Samples[i].Data, TimeSteps);

      valIdx := 1;
      for t := 0 to TimeSteps - 1 do
      begin
        SetLength(Result.Samples[i].Data[t], Features);
        for f := 0 to Features - 1 do
        begin
          Result.Samples[i].Data[t][f] := StrToFloatDef(tokens[valIdx], 0.0, fs);
          Inc(valIdx);
        end;
      end;
    end;
  finally
    tokens.Free;
    lines.Free;
  end;
end;

end.