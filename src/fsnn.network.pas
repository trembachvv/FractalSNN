unit fsnn.network;

{$mode objfpc}{$H+}
{$inline on}

interface

uses Math, fsnn.types, fsnn.rng, fsnn.matrix, fsnn.neuron, fsnn.datasets;

type
  TSNNNetwork = class
  private
    FConfig:      TExperimentConfig;
    FHidFractal:  TFractalNeuron;
    FOutFractal:  TFractalNeuron;
    AdamFracHid, AdamFracOut: array of TAdamState;
    AdamRecHid:   array of TAdamState;
    FSpikeTotal:  Int64;
    FActiveTotal: Int64;
    FSampleCount: Int64;
    FAccumCount:  Integer;
    FAccumHidW:   array of TMatrix;
    FAccumOutW:   array of TMatrix;
    FAccumRecHid: array of TMatrix;
    FClassWeights: array[0..1] of Double;
  public
    constructor Create(const Config: TExperimentConfig);
    destructor Destroy; override;
    function ForwardSample(const Sample: TSample): TDoubleArray;
    function TrainSample(const Sample: TSample; LR: Double): Double;
    procedure FlushAccum(LR: Double);
    function Evaluate(const D: TDataset): Double;
    procedure ResetState;
    procedure ResetStats;
    procedure PruneWeights(Fraction: Double);
    procedure QuantizeWeights(Bits: Integer);
    procedure QuantizeWeightsPerChannel(Bits: Integer);
    procedure ComputeClassWeights(const D: TDataset);
    property SpikeTotal: Int64 read FSpikeTotal;
    property ActiveTotal: Int64 read FActiveTotal;
    property SampleCount: Int64 read FSampleCount;
    property HidFractal: TFractalNeuron read FHidFractal;
    property Config: TExperimentConfig read FConfig;
    property ClassWeight0: Double read FClassWeights[0];
    property ClassWeight1: Double read FClassWeights[1];
  end;

implementation

constructor TSNNNetwork.Create(const Config: TExperimentConfig);
var l: Integer;
begin
  inherited Create;
  FConfig := Config;
  InitRNG(Config.Seed + Config.ThreadOffset);
  FHidFractal := TFractalNeuron.Create(
    Config.VThreshold, Config.Beta, Config.Surrogate,
    Config.FractalDepth, Config.SubNeuronsPerLvl,
    Config.InputSize, Config.HiddenSize, Config.Aggregation,
    Config.SpikeDropout, Config.KaimingGain,
    Config.Recurrent, Config.RecurrentScale,
    Config.TimeConstants, Config.BetaMin, Config.BetaMax,
    Config.LateralInhibition);
  FOutFractal := TFractalNeuron.Create(
    Config.VThreshold, Config.Beta, Config.Surrogate,
    1, 1, Config.HiddenSize, Config.OutputSize, agSum,
    0.0, Config.KaimingGain, False, 0.0,
    tcUniform, Config.Beta, Config.Beta, 0.0);

  SetLength(AdamFracHid, FHidFractal.DepthVal);
  SetLength(FAccumHidW, FHidFractal.DepthVal);
  SetLength(AdamRecHid, FHidFractal.DepthVal);
  SetLength(FAccumRecHid, FHidFractal.DepthVal);
  for l := 0 to FHidFractal.DepthVal - 1 do
  begin
    AdamFracHid[l] := AdamInit(MatCreate(
      FHidFractal.WeightRows(l), FHidFractal.WeightCols(l), 0.0));
    FAccumHidW[l] := MatCreate(
      FHidFractal.WeightRows(l), FHidFractal.WeightCols(l), 0.0);
    if Config.Recurrent then
    begin
      AdamRecHid[l] := AdamInit(MatCreate(
        FHidFractal.RecRows(l), FHidFractal.RecCols(l), 0.0));
      FAccumRecHid[l] := MatCreate(
        FHidFractal.RecRows(l), FHidFractal.RecCols(l), 0.0);
    end;
  end;

  SetLength(AdamFracOut, FOutFractal.DepthVal);
  SetLength(FAccumOutW, FOutFractal.DepthVal);
  for l := 0 to FOutFractal.DepthVal - 1 do
  begin
    AdamFracOut[l] := AdamInit(MatCreate(
      FOutFractal.WeightRows(l), FOutFractal.WeightCols(l), 0.0));
    FAccumOutW[l] := MatCreate(
      FOutFractal.WeightRows(l), FOutFractal.WeightCols(l), 0.0);
  end;

  FAccumCount := 0;
  FClassWeights[0] := 1.0;
  FClassWeights[1] := 1.0;
  ResetStats;
end;

destructor TSNNNetwork.Destroy;
begin
  if FHidFractal <> nil then FHidFractal.Free;
  if FOutFractal <> nil then FOutFractal.Free;
  inherited Destroy;
end;

procedure TSNNNetwork.ResetState;
begin
  FHidFractal.ResetState;
  FOutFractal.ResetState;
  FHidFractal.ClearHistory;
  FOutFractal.ClearHistory;
end;

procedure TSNNNetwork.ResetStats;
begin
  FSpikeTotal := 0;
  FActiveTotal := 0;
  FSampleCount := 0;
end;

procedure TSNNNetwork.PruneWeights(Fraction: Double);
begin
  FHidFractal.PruneWeights(Fraction);
  FOutFractal.PruneWeights(Fraction);
end;

procedure TSNNNetwork.QuantizeWeights(Bits: Integer);
begin
  FHidFractal.QuantizeWeights(Bits);
  FOutFractal.QuantizeWeights(Bits);
end;

procedure TSNNNetwork.QuantizeWeightsPerChannel(Bits: Integer);
begin
  FHidFractal.QuantizeWeightsPerChannel(Bits);
  FOutFractal.QuantizeWeightsPerChannel(Bits);
end;

procedure TSNNNetwork.ComputeClassWeights(const D: TDataset);
var i, c0, c1: Integer; total: Double;
begin
  c0 := 0; c1 := 0;
  for i := 0 to High(D.Samples) do
    if D.Samples[i].Label_ = 0 then Inc(c0) else Inc(c1);
  total := Length(D.Samples);
  if (c0 > 0) and (c1 > 0) then
  begin
    FClassWeights[0] := total / (2.0 * c0);
    FClassWeights[1] := total / (2.0 * c1);
  end
  else
  begin
    FClassWeights[0] := 1.0;
    FClassWeights[1] := 1.0;
  end;
end;

function TSNNNetwork.ForwardSample(const Sample: TSample): TDoubleArray;
var i, t: Integer; Input, Hid, Out_: TMatrix;
begin
  SetLength(Result, FConfig.OutputSize);
  for i := 0 to High(Result) do Result[i] := 0.0;
  ResetState;
  FHidFractal.ResetSpikeCount;
  FOutFractal.ResetSpikeCount;
  FHidFractal.SetTrainMode(False);
  FOutFractal.SetTrainMode(False);
  for t := 0 to FConfig.TimeSteps - 1 do
  begin
    Input := MatCreate(1, FConfig.InputSize);
    for i := 0 to FConfig.InputSize - 1 do
      Input.Data[i] := Sample.Data[t][i];
    Hid := FHidFractal.ForwardStep(Input);
    Out_ := FOutFractal.ForwardStep(Hid);
    for i := 0 to FConfig.OutputSize - 1 do
      Result[i] := Result[i] + Out_.Data[i];
  end;
  for i := 0 to FConfig.OutputSize - 1 do
    Result[i] := Result[i] / FConfig.TimeSteps;
  Inc(FSpikeTotal, FHidFractal.SpikeCount);
  Inc(FSpikeTotal, FOutFractal.SpikeCount);
  Inc(FActiveTotal, FHidFractal.ActiveNeurons);
  Inc(FSampleCount);
end;

procedure SoftmaxCE(const Logits: TDoubleArray; Target: Integer;
  ClassWeights: array of Double;
  out Probs: TDoubleArray; out Loss: Double); inline;
var i, n: Integer; mx, sum: Double; w: Double;
begin
  n := Length(Logits);
  SetLength(Probs, n);
  mx := Logits[0];
  for i := 1 to n - 1 do if Logits[i] > mx then mx := Logits[i];
  sum := 0.0;
  for i := 0 to n - 1 do
  begin
    Probs[i] := Exp(Logits[i] - mx);
    sum := sum + Probs[i];
  end;
  for i := 0 to n - 1 do Probs[i] := Probs[i] / sum;
  w := 1.0;
  if Target <= High(ClassWeights) then w := ClassWeights[Target];
  Loss := -w * Ln(Max(Probs[Target], 1e-12));
end;

function TSNNNetwork.TrainSample(const Sample: TSample; LR: Double): Double;
var
  i, t, l, idx, c: Integer;
  Input, Hid, Out_: TMatrix;
  logits, probs, dLogits: TDoubleArray;
  loss, classW, currentBeta: Double;
  gHidW, gOutW, gRecHid: array of TMatrix;
  dVHid: array of array of Double;
  dVOut, dSpikeOut, dSpikeHid, dPreOut, dSpikeL, dSpikeLm1: array of Double;
  dPreHid: array of array of Double;
begin
  FHidFractal.SetTrainMode(True);
  FOutFractal.SetTrainMode(True);
  ResetState;
  SetLength(logits, FConfig.OutputSize);
  for i := 0 to High(logits) do logits[i] := 0.0;
  for t := 0 to FConfig.TimeSteps - 1 do
  begin
    Input := MatCreate(1, FConfig.InputSize);
    for i := 0 to FConfig.InputSize - 1 do
      Input.Data[i] := Sample.Data[t][i];
    Hid := FHidFractal.ForwardStep(Input);
    Out_ := FOutFractal.ForwardStep(Hid);
    for i := 0 to FConfig.OutputSize - 1 do
      logits[i] := logits[i] + Out_.Data[i];
  end;
  for i := 0 to FConfig.OutputSize - 1 do
    logits[i] := logits[i] / FConfig.TimeSteps;
  SoftmaxCE(logits, Sample.Label_, FClassWeights, probs, loss);
  classW := 1.0;
  if Sample.Label_ <= 1 then classW := FClassWeights[Sample.Label_];
  SetLength(dLogits, FConfig.OutputSize);
  for i := 0 to High(dLogits) do
    dLogits[i] := classW * probs[i] / FConfig.TimeSteps;
  dLogits[Sample.Label_] := dLogits[Sample.Label_] - classW / FConfig.TimeSteps;

  SetLength(gHidW, FHidFractal.DepthVal);
  SetLength(gRecHid, FHidFractal.DepthVal);
  for l := 0 to FHidFractal.DepthVal - 1 do
  begin
    gHidW[l] := MatCreate(FHidFractal.WeightRows(l),
                          FHidFractal.WeightCols(l), 0.0);
    if FConfig.Recurrent then
      gRecHid[l] := MatCreate(FHidFractal.RecRows(l),
                              FHidFractal.RecCols(l), 0.0)
    else
      gRecHid[l] := MatCreate(0, 0);
  end;
  SetLength(gOutW, FOutFractal.DepthVal);
  for l := 0 to FOutFractal.DepthVal - 1 do
    gOutW[l] := MatCreate(FOutFractal.WeightRows(l),
                          FOutFractal.WeightCols(l), 0.0);

  SetLength(dVHid, FHidFractal.DepthVal);
  for l := 0 to FHidFractal.DepthVal - 1 do
    SetLength(dVHid[l], FHidFractal.SubVal * FHidFractal.OutSize);
  SetLength(dVOut, FOutFractal.SubVal * FOutFractal.OutSize);

  for t := FConfig.TimeSteps - 1 downto 0 do
  begin
    SetLength(dSpikeOut, FOutFractal.SubVal * FOutFractal.OutSize);
    for idx := 0 to High(dSpikeOut) do dSpikeOut[idx] := 0.0;
    for c := 0 to FOutFractal.OutSize - 1 do dSpikeOut[c] := dLogits[c];
    SetLength(dPreOut, FOutFractal.SubVal * FOutFractal.OutSize);
    for idx := 0 to High(dPreOut) do
      dPreOut[idx] := dSpikeOut[idx] *
        FOutFractal.SurrogateGrad(FOutFractal.HistV(t, 0, idx)) + dVOut[idx];
    for i := 0 to FOutFractal.WeightRows(0) - 1 do
      for idx := 0 to FOutFractal.WeightCols(0) - 1 do
        gOutW[0].Data[i * FOutFractal.WeightCols(0) + idx] :=
          gOutW[0].Data[i * FOutFractal.WeightCols(0) + idx]
          + FHidFractal.HistSpike(t, 0, i) * dPreOut[idx];
    SetLength(dSpikeHid, FHidFractal.SubVal * FHidFractal.OutSize);
    for i := 0 to High(dSpikeHid) do dSpikeHid[i] := 0.0;
    for i := 0 to FOutFractal.WeightRows(0) - 1 do
      for c := 0 to FOutFractal.WeightCols(0) - 1 do
        dSpikeHid[i] := dSpikeHid[i]
          + dPreOut[c] * FOutFractal.WeightAt(0, i * FOutFractal.WeightCols(0) + c);
    for idx := 0 to High(dVOut) do
      dVOut[idx] := dPreOut[idx] * FConfig.Beta;

    SetLength(dPreHid, FHidFractal.DepthVal);
    for l := 0 to FHidFractal.DepthVal - 1 do
      SetLength(dPreHid[l], FHidFractal.SubVal * FHidFractal.OutSize);
    l := FHidFractal.DepthVal - 1;
    for idx := 0 to High(dPreHid[l]) do
      dPreHid[l][idx] :=
        (dSpikeHid[idx] / FHidFractal.SubVal) *
        FHidFractal.SurrogateGrad(FHidFractal.HistV(t, l, idx)) + dVHid[l][idx];
    for i := 0 to FHidFractal.WeightRows(l) - 1 do
      for idx := 0 to FHidFractal.WeightCols(l) - 1 do
      begin
        if l = 0 then
          gHidW[l].Data[i * FHidFractal.WeightCols(l) + idx] :=
            gHidW[l].Data[i * FHidFractal.WeightCols(l) + idx]
            + Sample.Data[t][i] * dPreHid[l][idx]
        else
          gHidW[l].Data[i * FHidFractal.WeightCols(l) + idx] :=
            gHidW[l].Data[i * FHidFractal.WeightCols(l) + idx]
            + FHidFractal.HistSpike(t, l - 1, i) * dPreHid[l][idx];
      end;
    SetLength(dSpikeLm1, FHidFractal.SubVal * FHidFractal.OutSize);
    for i := 0 to High(dSpikeLm1) do dSpikeLm1[i] := 0.0;
    for i := 0 to FHidFractal.WeightRows(l) - 1 do
      for idx := 0 to FHidFractal.WeightCols(l) - 1 do
        dSpikeLm1[i] := dSpikeLm1[i]
          + dPreHid[l][idx] *
            FHidFractal.WeightAt(l, i * FHidFractal.WeightCols(l) + idx);
    if FHidFractal.DepthVal >= 2 then
      for l := FHidFractal.DepthVal - 2 downto 0 do
      begin
        for idx := 0 to High(dPreHid[l]) do
          dPreHid[l][idx] := dSpikeLm1[idx] *
            FHidFractal.SurrogateGrad(FHidFractal.HistV(t, l, idx)) + dVHid[l][idx];
        for i := 0 to FHidFractal.WeightRows(l) - 1 do
          for idx := 0 to FHidFractal.WeightCols(l) - 1 do
          begin
            if l = 0 then
              gHidW[l].Data[i * FHidFractal.WeightCols(l) + idx] :=
                gHidW[l].Data[i * FHidFractal.WeightCols(l) + idx]
                + Sample.Data[t][i] * dPreHid[l][idx]
            else
              gHidW[l].Data[i * FHidFractal.WeightCols(l) + idx] :=
                gHidW[l].Data[i * FHidFractal.WeightCols(l) + idx]
                + FHidFractal.HistSpike(t, l - 1, i) * dPreHid[l][idx];
          end;
        if l > 0 then
        begin
          SetLength(dSpikeL, FHidFractal.SubVal * FHidFractal.OutSize);
          for i := 0 to High(dSpikeL) do dSpikeL[i] := 0.0;
          for i := 0 to FHidFractal.WeightRows(l) - 1 do
            for idx := 0 to FHidFractal.WeightCols(l) - 1 do
              dSpikeL[i] := dSpikeL[i]
                + dPreHid[l][idx] *
                  FHidFractal.WeightAt(l, i * FHidFractal.WeightCols(l) + idx);
          dSpikeLm1 := dSpikeL;
        end;
      end;
    for l := 0 to FHidFractal.DepthVal - 1 do
    begin
      currentBeta := FHidFractal.BetaAt(l);
      for idx := 0 to High(dVHid[l]) do
        dVHid[l][idx] := dPreHid[l][idx] * currentBeta;
    end;
  end;

  for l := 0 to FHidFractal.DepthVal - 1 do
  begin
    for i := 0 to High(gHidW[l].Data) do
      FAccumHidW[l].Data[i] := FAccumHidW[l].Data[i] + gHidW[l].Data[i];
    if FConfig.Recurrent then
      for i := 0 to High(gRecHid[l].Data) do
        FAccumRecHid[l].Data[i] := FAccumRecHid[l].Data[i] + gRecHid[l].Data[i];
  end;
  for l := 0 to FOutFractal.DepthVal - 1 do
    for i := 0 to High(gOutW[l].Data) do
      FAccumOutW[l].Data[i] := FAccumOutW[l].Data[i] + gOutW[l].Data[i];
  Inc(FAccumCount);
  if FAccumCount >= FConfig.BatchAccum then FlushAccum(LR);
  Result := loss;
end;

procedure TSNNNetwork.FlushAccum(LR: Double);
var
  l, i: Integer;
  scale, sumg: Double;
  effWD, effLR: Double;
begin
  if FAccumCount = 0 then Exit;
  scale := 1.0 / FAccumCount;
  for l := 0 to FHidFractal.DepthVal - 1 do
  begin
    for i := 0 to High(FAccumHidW[l].Data) do
      FAccumHidW[l].Data[i] := FAccumHidW[l].Data[i] * scale;
    sumg := MatSum(FAccumHidW[l]);
    if Abs(sumg) > EPS then
    begin
      // ТЗ Баг 4: Раздельное затухание весов и LR для нижних и верхних слоев
      if l >= FHidFractal.DepthVal - 2 then
      begin
        effWD := FConfig.WeightDecay;
        effLR := LR;
      end
      else
      begin
        effWD := FConfig.WeightDecay * 0.1;
        effLR := LR * 0.5;
      end;

      FHidFractal.ApplyGrad(l, FAccumHidW[l], AdamFracHid[l],
        effLR, FConfig.Beta1, FConfig.Beta2,
        FConfig.AdamEps, effWD, FConfig.GradientClip);
    end;
    for i := 0 to High(FAccumHidW[l].Data) do FAccumHidW[l].Data[i] := 0.0;
    if FConfig.Recurrent then
    begin
      for i := 0 to High(FAccumRecHid[l].Data) do
        FAccumRecHid[l].Data[i] := FAccumRecHid[l].Data[i] * scale;
      sumg := MatSum(FAccumRecHid[l]);
      if Abs(sumg) > EPS then
        FHidFractal.ApplyRecGrad(l, FAccumRecHid[l], AdamRecHid[l],
          LR, FConfig.Beta1, FConfig.Beta2,
          FConfig.AdamEps, FConfig.WeightDecay, FConfig.GradientClip);
      for i := 0 to High(FAccumRecHid[l].Data) do
        FAccumRecHid[l].Data[i] := 0.0;
    end;
  end;
  for l := 0 to FOutFractal.DepthVal - 1 do
  begin
    for i := 0 to High(FAccumOutW[l].Data) do
      FAccumOutW[l].Data[i] := FAccumOutW[l].Data[i] * scale;
    FOutFractal.ApplyGrad(l, FAccumOutW[l], AdamFracOut[l],
      LR, FConfig.Beta1, FConfig.Beta2,
      FConfig.AdamEps, FConfig.WeightDecay, FConfig.GradientClip);
    for i := 0 to High(FAccumOutW[l].Data) do FAccumOutW[l].Data[i] := 0.0;
  end;
  FAccumCount := 0;
end;

function TSNNNetwork.Evaluate(const D: TDataset): Double;
var i, j, pred, correct: Integer; logits: TDoubleArray;
begin
  correct := 0;
  FHidFractal.SetTrainMode(False);
  FOutFractal.SetTrainMode(False);
  for i := 0 to High(D.Samples) do
  begin
    logits := ForwardSample(D.Samples[i]);
    pred := 0;
    for j := 1 to High(logits) do
      if logits[j] > logits[pred] then pred := j;
    if pred = D.Samples[i].Label_ then Inc(correct);
  end;
  if Length(D.Samples) > 0 then Result := correct / Length(D.Samples)
  else Result := 0.0;
end;

end.