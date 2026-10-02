unit fsnn.train;

{$mode objfpc}{$H+}

interface

uses
  Math, SysUtils,
  fsnn.types, fsnn.rng, fsnn.matrix, fsnn.network, fsnn.datasets;

procedure TrainNetwork(Net: TSNNNetwork; const TrainSet, TestSet: TDataset;
  const Cfg: TExperimentConfig; out Res: TRunResult; Verbose: Boolean;
  LogProc: TLogProc = nil);

function EvaluateMetrics(Net: TSNNNetwork; const D: TDataset;
  out Prec, Rec, F1, Pred0Rate, Pred1Rate: Double): Double;

implementation

procedure TrainNetwork(Net: TSNNNetwork; const TrainSet, TestSet: TDataset;
  const Cfg: TExperimentConfig; out Res: TRunResult; Verbose: Boolean;
  LogProc: TLogProc = nil);
var
  epoch, i, j, tmp: Integer;
  totalLoss, avgLoss, testAcc, bestAcc: Double;
  patience: Integer;
  startTime: TDateTime;
  order: array of Integer;
  lrNow: Double;
  prec, rec, f1, p0, p1: Double;
begin
  startTime := Now;
  bestAcc := 0.0;
  patience := 0;
  avgLoss := 0.0;
  epoch := 0;

  Res.TestAccuracy := 0.0;
  Res.TestF1 := 0.0;
  Res.TestPrecision := 0.0;
  Res.TestRecall := 0.0;
  Res.TrainAccuracy := 0.0;
  Res.MeanSpikeCount := 0.0;
  Res.MeanActiveNrn := 0.0;
  Res.EnergyPJ := 0.0;
  Res.TotalParams := 0;
  Res.TrainSeconds := 0.0;
  Res.EpochsRun := 0;
  Res.FinalLoss := 0.0;
  Res.Class0PredRate := 0.0;
  Res.Class1PredRate := 0.0;
  Res.Collapsed := False;

  if Cfg.ClassBalance = cbWeights then
    Net.ComputeClassWeights(TrainSet);

  SetLength(order, Length(TrainSet.Samples));
  for i := 0 to High(order) do order[i] := i;

  for epoch := 1 to Cfg.Epochs do
  begin
    if epoch <= Cfg.WarmupEpochs then
      lrNow := Cfg.LearningRate0 * epoch / Cfg.WarmupEpochs
    else
      lrNow := Cfg.LearningRate0 * Power(Cfg.LearningRate1 / Cfg.LearningRate0,
        (epoch - Cfg.WarmupEpochs) / Max(1, Cfg.Epochs - Cfg.WarmupEpochs));

    // Изолированное потокобезопасное перемешивание порядка сэмплов
    for i := High(order) downto 1 do
    begin
      j := RandIntRange(i + 1);
      tmp := order[i]; order[i] := order[j]; order[j] := tmp;
    end;

    totalLoss := 0.0;
    for i := 0 to High(order) do
      totalLoss := totalLoss + Net.TrainSample(TrainSet.Samples[order[i]], lrNow);
    Net.FlushAccum(lrNow);
    avgLoss := totalLoss / Length(order);
    testAcc := Net.Evaluate(TestSet);

    if Verbose and Assigned(LogProc) then
      LogProc(Format(' epoch %3d | lr=%.5f | loss=%.4f | testAcc=%.4f',
        [epoch, lrNow, avgLoss, testAcc]));

    if testAcc > bestAcc then
    begin
      bestAcc := testAcc;
      patience := 0;
    end
    else
      Inc(patience);
    if patience >= Cfg.EarlyStopPatience then
    begin
      if Verbose and Assigned(LogProc) then
        LogProc(' early stop at epoch ' + IntToStr(epoch));
      Break;
    end;
  end;

  Net.ResetStats;
  Res.TestAccuracy := EvaluateMetrics(Net, TestSet, prec, rec, f1, p0, p1);
  Res.TestF1 := f1;
  Res.TestPrecision := prec;
  Res.TestRecall := rec;
  Res.Class0PredRate := p0;
  Res.Class1PredRate := p1;
  Res.Collapsed := (p0 > COLLAPSE_THRESHOLD) or (p1 > COLLAPSE_THRESHOLD);
  Res.TrainAccuracy := Net.Evaluate(TrainSet);
  Res.MeanSpikeCount := Net.SpikeTotal / Max(1, Net.SampleCount);
  Res.MeanActiveNrn := Net.ActiveTotal / Max(1, Net.SampleCount);
  Res.EnergyPJ := Res.MeanSpikeCount * E_SYNAPSE_PJ +
                  Res.MeanActiveNrn * Cfg.TimeSteps * E_MEMBRANE_PJ;
  Res.TotalParams := Net.HidFractal.ParamCount;
  Res.TrainSeconds := (Now - startTime) * SecsPerDay;
  Res.EpochsRun := epoch;
  Res.FinalLoss := avgLoss;
end;

function EvaluateMetrics(Net: TSNNNetwork; const D: TDataset;
  out Prec, Rec, F1, Pred0Rate, Pred1Rate: Double): Double;
var
  i, j, pred, TP, FP, FN, TN, cnt0, cnt1: Integer;
  logits: TDoubleArray;
begin
  TP := 0; FP := 0; FN := 0; TN := 0; cnt0 := 0; cnt1 := 0;
  for i := 0 to High(D.Samples) do
  begin
    logits := Net.ForwardSample(D.Samples[i]);
    pred := 0;
    for j := 1 to High(logits) do
      if logits[j] > logits[pred] then pred := j;
    if pred = 0 then Inc(cnt0) else Inc(cnt1);
    if (pred = 1) and (D.Samples[i].Label_ = 1) then Inc(TP)
    else if (pred = 1) and (D.Samples[i].Label_ = 0) then Inc(FP)
    else if (pred = 0) and (D.Samples[i].Label_ = 1) then Inc(FN)
    else Inc(TN);
  end;
  if TP + FP > 0 then Prec := TP / (TP + FP) else Prec := 0;
  if TP + FN > 0 then Rec := TP / (TP + FN) else Rec := 0;
  if Prec + Rec > 0 then F1 := 2 * Prec * Rec / (Prec + Rec) else F1 := 0;
  if Length(D.Samples) > 0 then
  begin
    Result := (TP + TN) / Length(D.Samples);
    Pred0Rate := cnt0 / Length(D.Samples);
    Pred1Rate := cnt1 / Length(D.Samples);
  end
  else
  begin
    Result := 0; Pred0Rate := 0; Pred1Rate := 0;
  end;
end;

end.