unit fsnn.threads;

{$mode objfpc}{$H+}

interface

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  fsnn.types, fsnn.rng, fsnn.datasets,
  fsnn.neuron, fsnn.network, fsnn.train, fsnn.log;

type
  PSeedJob = ^TSeedJob;
  TSeedJob = record
    Config: TExperimentConfig;
    Seed: Integer;
    ThreadIdx: Integer;
    Result_: TRunResult;
    R: TRng;
    Success: Boolean;
  end;

  TSeedWorker = class(TThread)
  private
    FJob: PSeedJob;
    FLogProc: TLogProc;
  protected
    procedure Execute; override;
  public
    constructor Create(AJob: PSeedJob; ALogProc: TLogProc);
  end;

procedure RunSeedsParallel(const Cfg: TExperimentConfig;
  const Seeds: array of Integer;
  NumThreads: Integer;
  out Results: TRunResultArray;
  LogProc: TLogProc = nil);

implementation

constructor TSeedWorker.Create(AJob: PSeedJob; ALogProc: TLogProc);
begin
  inherited Create(True);
  FJob := AJob;
  FLogProc := ALogProc;
  FreeOnTerminate := False;
end;

procedure TSeedWorker.Execute;
var
  net: TSNNNetwork;
  trainSet, testSet: TDataset;
  localCfg: TExperimentConfig;
begin
  FJob^.Success := False;
  try
    localCfg := FJob^.Config;
    localCfg.Seed := FJob^.Seed;
    localCfg.ThreadOffset := FJob^.ThreadIdx;

    // Инициализация собственного RNG строго через Seed + ThreadOffset
    InitRNG((localCfg.Seed + localCfg.ThreadOffset) * 1000 + 1);

    trainSet := GenerateDataset(localCfg, localCfg.TrainSamples, localCfg.Seed);
    testSet := GenerateDataset(localCfg, localCfg.TestSamples, localCfg.Seed + 1000);

    net := TSNNNetwork.Create(localCfg);
    try
      TrainNetwork(net, trainSet, testSet, localCfg, FJob^.Result_, False, FLogProc);
    finally
      net.Free;
    end;

    FJob^.Success := True;
  except
    on E: Exception do
    begin
      if Assigned(FLogProc) then
        FLogProc(Format('[T%d] EXCEPTION seed=%d: %s: %s',
          [FJob^.ThreadIdx, FJob^.Seed, E.ClassName, E.Message]));
      FJob^.Success := False;
      FJob^.Result_.TestAccuracy := 0.0;
      FJob^.Result_.Collapsed := True;
    end;
  end;
end;

procedure RunSeedsParallel(const Cfg: TExperimentConfig;
  const Seeds: array of Integer;
  NumThreads: Integer;
  out Results: TRunResultArray;
  LogProc: TLogProc = nil);
var
  Jobs: array of TSeedJob;
  Workers: array of TSeedWorker;
  i, j, n, batchStart, batchEnd, batchSize: Integer;
  SingleTrain, SingleTest: TDataset;
  SingleNet: TSNNNetwork;
  SingleCfg: TExperimentConfig;
begin
  n := Length(Seeds);
  SetLength(Results, n);
  if n = 0 then Exit;

  // Однопоточный путь: передаем ровно такой же ThreadOffset := i для детерминизма
  if (NumThreads <= 1) or (n = 1) then
  begin
    for i := 0 to n - 1 do
    begin
      SingleCfg := Cfg;
      SingleCfg.Seed := Seeds[i];
      SingleCfg.ThreadOffset := i;
      InitRNG((Seeds[i] + SingleCfg.ThreadOffset) * 1000 + 1);
      SingleTrain := GenerateDataset(SingleCfg, SingleCfg.TrainSamples, Seeds[i]);
      SingleTest := GenerateDataset(SingleCfg, SingleCfg.TestSamples, Seeds[i] + 1000);
      SingleNet := TSNNNetwork.Create(SingleCfg);
      try
        TrainNetwork(SingleNet, SingleTrain, SingleTest, SingleCfg, Results[i], False, LogProc);
      finally
        SingleNet.Free;
      end;
    end;
    Exit;
  end;

  // Многопоточный пул
  SetLength(Jobs, n);
  for i := 0 to n - 1 do
  begin
    Jobs[i].Config := Cfg;
    Jobs[i].Seed := Seeds[i];
    Jobs[i].ThreadIdx := i;
    FillChar(Jobs[i].Result_, SizeOf(TRunResult), 0);
    Jobs[i].Success := False;
  end;

  batchStart := 0;
  while batchStart < n do
  begin
    batchEnd := batchStart + NumThreads - 1;
    if batchEnd > n - 1 then batchEnd := n - 1;
    batchSize := batchEnd - batchStart + 1;

    SetLength(Workers, batchSize);
    for j := 0 to batchSize - 1 do
    begin
      Workers[j] := TSeedWorker.Create(@Jobs[batchStart + j], LogProc);
      Workers[j].Start;
    end;

    for j := 0 to batchSize - 1 do
    begin
      Workers[j].WaitFor;
      Workers[j].Free;
    end;

    for j := 0 to batchSize - 1 do
      Results[batchStart + j] := Jobs[batchStart + j].Result_;

    batchStart := batchEnd + 1;
  end;
end;

end.