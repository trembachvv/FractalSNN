unit fsnn.datasets;

{$mode objfpc}{$H+}

interface

uses Math, fsnn.types, fsnn.rng;

type
  TTimeStep = array of Double;
  TSample = record
    Data:  array of TTimeStep;
    Label_: Integer;
  end;
  TDataset = record
    Samples:    array of TSample;
    InputSize:  Integer;
    TimeSteps:  Integer;
    NumClasses: Integer;
  end;

function GenerateTemporalXORDataset(NumSamples, TSteps, InputSize: Integer;
  Seed: Integer; NoiseSigma: Double; DelaySteps: Integer): TDataset;
function GenerateDelayedMatchDataset(NumSamples, TSteps, InputSize: Integer;
  Seed: Integer; NoiseSigma: Double; DelaySteps: Integer): TDataset;
function GenerateSpokenLikeDataset(NumSamples, TSteps, InputSize: Integer;
  Seed: Integer; NoiseSigma: Double): TDataset;
function GeneratePatternRecogDataset(NumSamples, TSteps, InputSize: Integer;
  Seed: Integer; NoiseSigma: Double): TDataset;
function GenerateSequenceMemDataset(NumSamples, TSteps, InputSize: Integer;
  Seed: Integer; NoiseSigma: Double): TDataset;
function GenerateDataset(const Cfg: TExperimentConfig;
  NumSamples: Integer; Seed: Integer): TDataset;

implementation

function GenerateTemporalXORDataset(NumSamples, TSteps, InputSize: Integer;
  Seed: Integer; NoiseSigma: Double; DelaySteps: Integer): TDataset;
var
  i, t, c, targetLabel, posA, posB: Integer;
  bitA, bitB: Integer;
  spikeA, spikeB: Boolean;
begin
  InitRNG(Seed);
  Result.InputSize := InputSize;
  Result.TimeSteps := TSteps;
  Result.NumClasses := 2;
  SetLength(Result.Samples, NumSamples);
  for i := 0 to NumSamples - 1 do
  begin
    SetLength(Result.Samples[i].Data, TSteps);
    for t := 0 to TSteps - 1 do
    begin
      SetLength(Result.Samples[i].Data[t], InputSize);
      for c := 0 to InputSize - 1 do
        Result.Samples[i].Data[t][c] := RandGaussian(NoiseSigma);
    end;
    bitA := i mod 2;
    bitB := (i div 2) mod 2;
    targetLabel := bitA xor bitB;
    spikeA := bitA = 1;
    spikeB := bitB = 1;
    posA := 2 + RandIntRange(3);
    posB := posA + DelaySteps + RandIntRange(3);
    if spikeA and (posA < TSteps) then
    begin
      Result.Samples[i].Data[posA][0] := 1.5 + RandGaussian(NoiseSigma);
      if posA + 1 < TSteps then
        Result.Samples[i].Data[posA + 1][0] := 1.0 + RandGaussian(NoiseSigma);
    end;
    if spikeB and (posB < TSteps) then
    begin
      Result.Samples[i].Data[posB][1] := 1.5 + RandGaussian(NoiseSigma);
      if posB + 1 < TSteps then
        Result.Samples[i].Data[posB + 1][1] := 1.0 + RandGaussian(NoiseSigma);
    end;
    Result.Samples[i].Label_ := targetLabel;
  end;
end;

function GenerateDelayedMatchDataset(NumSamples, TSteps, InputSize: Integer;
  Seed: Integer; NoiseSigma: Double; DelaySteps: Integer): TDataset;
var i, t, c, stimA, stimB: Integer;
begin
  InitRNG(Seed);
  Result.InputSize := InputSize;
  Result.TimeSteps := TSteps;
  Result.NumClasses := 2;
  SetLength(Result.Samples, NumSamples);
  for i := 0 to NumSamples - 1 do
  begin
    SetLength(Result.Samples[i].Data, TSteps);
    for t := 0 to TSteps - 1 do
    begin
      SetLength(Result.Samples[i].Data[t], InputSize);
      for c := 0 to InputSize - 1 do
        Result.Samples[i].Data[t][c] := RandGaussian(NoiseSigma);
    end;
    stimA := RandIntRange(2);
    stimB := RandIntRange(2);
    if stimA = 1 then
    begin
      Result.Samples[i].Data[1][0] := 1.5 + RandGaussian(NoiseSigma);
      Result.Samples[i].Data[2][0] := 1.0 + RandGaussian(NoiseSigma);
    end;
    if DelaySteps + 4 < TSteps then
    begin
      if stimB = 1 then
      begin
        Result.Samples[i].Data[DelaySteps + 3][1] := 1.5 + RandGaussian(NoiseSigma);
        Result.Samples[i].Data[DelaySteps + 4][1] := 1.0 + RandGaussian(NoiseSigma);
      end;
    end;
    if stimA = stimB then Result.Samples[i].Label_ := 1
    else Result.Samples[i].Label_ := 0;
  end;
end;

function GenerateSpokenLikeDataset(NumSamples, TSteps, InputSize: Integer;
  Seed: Integer; NoiseSigma: Double): TDataset;
var i, t, c, classLabel: Integer; freq: Double;
begin
  InitRNG(Seed);
  Result.InputSize := InputSize;
  Result.TimeSteps := TSteps;
  Result.NumClasses := 2;
  SetLength(Result.Samples, NumSamples);
  for i := 0 to NumSamples - 1 do
  begin
    SetLength(Result.Samples[i].Data, TSteps);
    classLabel := i mod 2;
    if classLabel = 0 then freq := 0.3 else freq := 0.7;
    for t := 0 to TSteps - 1 do
    begin
      SetLength(Result.Samples[i].Data[t], InputSize);
      for c := 0 to InputSize - 1 do
        Result.Samples[i].Data[t][c] := RandGaussian(NoiseSigma);
      if Sin(2 * Pi * freq * t / TSteps) > 0.5 then
        Result.Samples[i].Data[t][0] := 1.0 + RandGaussian(NoiseSigma);
    end;
    Result.Samples[i].Label_ := classLabel;
  end;
end;

function GeneratePatternRecogDataset(NumSamples, TSteps, InputSize: Integer;
  Seed: Integer; NoiseSigma: Double): TDataset;
var
  i, t, c, classLabel, patStart, patLen: Integer;
  pattern: array of Integer;
begin
  InitRNG(Seed);
  Result.InputSize := InputSize;
  Result.TimeSteps := TSteps;
  Result.NumClasses := 2;
  SetLength(Result.Samples, NumSamples);
  SetLength(pattern, 4);
  for i := 0 to NumSamples - 1 do
  begin
    SetLength(Result.Samples[i].Data, TSteps);
    classLabel := i mod 2;
    if classLabel = 0 then
    begin
      pattern[0] := 0; pattern[1] := 1; pattern[2] := 0; pattern[3] := 1;
    end
    else
    begin
      pattern[0] := 1; pattern[1] := 0; pattern[2] := 1; pattern[3] := 0;
    end;
    patLen := 4;
    patStart := 2 + RandIntRange(Max(1, TSteps - patLen - 4));
    for t := 0 to TSteps - 1 do
    begin
      SetLength(Result.Samples[i].Data[t], InputSize);
      for c := 0 to InputSize - 1 do
        Result.Samples[i].Data[t][c] := RandGaussian(NoiseSigma);
      if (t >= patStart) and (t < patStart + patLen) then
      begin
        if pattern[t - patStart] = 1 then
          Result.Samples[i].Data[t][0] := 1.5 + RandGaussian(NoiseSigma);
      end;
    end;
    Result.Samples[i].Label_ := classLabel;
  end;
end;

function GenerateSequenceMemDataset(NumSamples, TSteps, InputSize: Integer;
  Seed: Integer; NoiseSigma: Double): TDataset;
var
  i, t, c, classLabel, firstPos, secondPos: Integer;
  orderA, orderB: Integer;
begin
  InitRNG(Seed);
  Result.InputSize := InputSize;
  Result.TimeSteps := TSteps;
  Result.NumClasses := 2;
  SetLength(Result.Samples, NumSamples);
  for i := 0 to NumSamples - 1 do
  begin
    SetLength(Result.Samples[i].Data, TSteps);
    for t := 0 to TSteps - 1 do
    begin
      SetLength(Result.Samples[i].Data[t], InputSize);
      for c := 0 to InputSize - 1 do
        Result.Samples[i].Data[t][c] := RandGaussian(NoiseSigma);
    end;
    classLabel := i mod 2;
    if classLabel = 0 then begin orderA := 0; orderB := 1; end
    else begin orderA := 1; orderB := 0; end;
    firstPos := 2 + RandIntRange(3);
    secondPos := firstPos + 3 + RandIntRange(3);
    if secondPos < TSteps then
    begin
      Result.Samples[i].Data[firstPos][orderA] := 1.5 + RandGaussian(NoiseSigma);
      Result.Samples[i].Data[secondPos][orderB] := 1.5 + RandGaussian(NoiseSigma);
    end;
    Result.Samples[i].Label_ := classLabel;
  end;
end;

function GenerateDataset(const Cfg: TExperimentConfig;
  NumSamples: Integer; Seed: Integer): TDataset;
begin
  case Cfg.Dataset of
    dkTemporalXOR:
      Result := GenerateTemporalXORDataset(NumSamples, Cfg.TimeSteps,
        Cfg.InputSize, Seed, Cfg.NoiseSigma, Cfg.DelaySteps);
    dkDelayedMatch:
      Result := GenerateDelayedMatchDataset(NumSamples, Cfg.TimeSteps,
        Cfg.InputSize, Seed, Cfg.NoiseSigma, Cfg.DelaySteps);
    dkSpokenLike:
      Result := GenerateSpokenLikeDataset(NumSamples, Cfg.TimeSteps,
        Cfg.InputSize, Seed, Cfg.NoiseSigma);
    dkPatternRecog:
      Result := GeneratePatternRecogDataset(NumSamples, Cfg.TimeSteps,
        Cfg.InputSize, Seed, Cfg.NoiseSigma);
    dkSequenceMem:
      Result := GenerateSequenceMemDataset(NumSamples, Cfg.TimeSteps,
        Cfg.InputSize, Seed, Cfg.NoiseSigma);
  else
    FillChar(Result, SizeOf(Result), 0);
  end;
end;

end.