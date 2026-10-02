unit fsnn.lif;

{$mode objfpc}{$H+}

interface

uses
  Math, SysUtils,
  fsnn.types, fsnn.rng, fsnn.matrix, fsnn.datasets;

type
  TLIFNetwork = class
  private
    FConfig: TExperimentConfig;
    FHiddenSize: Integer;
    FInputSize: Integer;
    FOutputSize: Integer;

    // Веса и смещения
    FWHid: array of Double;      // [InputSize * HiddenSize]
    FWOut: array of Double;      // [HiddenSize * OutputSize]

    // Состояния Adam для скрытого и выходного слоев
    FMHid, FVmHid: array of Double;
    FMOut, FVmOut: array of Double;
    FTAdam: Int64;

    // Мембранные потенциалы
    FVHid: array of Double;
    FVOut: array of Double;

    // Статистика
    FSpikeTotal: Int64;
    FSampleCount: Int64;
    FClassWeights: array[0..1] of Double;

    // Буферы BPTT
    FSpikeHidHist: array of array of Double;
    FVHidHist: array of array of Double;
    FGHid: array of Double;
    FGOut: array of Double;
    FDSpikeHid: array of Double;
    FDPreHid: array of Double;
    FLogits: array of Double;
    FProbs: array of Double;
    FDLogits: array of Double;
    FAllocatedT: Integer;
  public
    constructor Create(const Config: TExperimentConfig; MatchingFractalParams: Integer);
    destructor Destroy; override;
    function ForwardSample(const Sample: TSample): TDoubleArray;
    function TrainSample(const Sample: TSample; LR: Double): Double;
    function Evaluate(const D: TDataset): Double;
    procedure ResetState;
    procedure ResetStats;
    procedure ComputeClassWeights(const D: TDataset);
    property SpikeTotal: Int64 read FSpikeTotal;
    property SampleCount: Int64 read FSampleCount;
    property HiddenSize: Integer read FHiddenSize;
    property ClassWeight0: Double read FClassWeights[0];
    property ClassWeight1: Double read FClassWeights[1];
    function ParamCount: Integer;
  end;

function CalculateLIFHiddenSize(FractalParams, InSize, OutSize: Integer): Integer;

implementation

function CalculateLIFHiddenSize(FractalParams, InSize, OutSize: Integer): Integer;
begin
  Result := Round(FractalParams / Max(1, InSize + OutSize));
  if Result < 1 then Result := 1;
end;

constructor TLIFNetwork.Create(const Config: TExperimentConfig; MatchingFractalParams: Integer);
var
  i, totalHidW, totalOutW, t: Integer;
  scaleHid, scaleOut: Double;
begin
  inherited Create;
  FConfig := Config;
  FInputSize := Max(1, Config.InputSize);
  FOutputSize := Max(1, Config.OutputSize);
  FHiddenSize := CalculateLIFHiddenSize(MatchingFractalParams, FInputSize, FOutputSize);

  totalHidW := FInputSize * FHiddenSize;
  totalOutW := FHiddenSize * FOutputSize;

  SetLength(FWHid, totalHidW);
  SetLength(FWOut, totalOutW);
  SetLength(FMHid, totalHidW);
  SetLength(FVmHid, totalHidW);
  SetLength(FMOut, totalOutW);
  SetLength(FVmOut, totalOutW);
  FTAdam := 0;

  // Инициализация Kaiming
  scaleHid := Config.KaimingGain * Sqrt(2.0 / FInputSize);
  for i := 0 to totalHidW - 1 do
  begin
    FWHid[i] := RandGaussian(scaleHid);
    FMHid[i] := 0.0;
    FVmHid[i] := 0.0;
  end;

  scaleOut := Config.KaimingGain * Sqrt(2.0 / FHiddenSize);
  for i := 0 to totalOutW - 1 do
  begin
    FWOut[i] := RandGaussian(scaleOut);
    FMOut[i] := 0.0;
    FVmOut[i] := 0.0;
  end;

  SetLength(FVHid, FHiddenSize);
  SetLength(FVOut, FOutputSize);

  FAllocatedT := Max(1, Config.TimeSteps);
  SetLength(FSpikeHidHist, FAllocatedT);
  SetLength(FVHidHist, FAllocatedT);
  for t := 0 to FAllocatedT - 1 do
  begin
    SetLength(FSpikeHidHist[t], FHiddenSize);
    SetLength(FVHidHist[t], FHiddenSize);
  end;

  SetLength(FGHid, totalHidW);
  SetLength(FGOut, totalOutW);
  SetLength(FDSpikeHid, FHiddenSize);
  SetLength(FDPreHid, FHiddenSize);
  SetLength(FLogits, FOutputSize);
  SetLength(FProbs, FOutputSize);
  SetLength(FDLogits, FOutputSize);

  FClassWeights[0] := 1.0;
  FClassWeights[1] := 1.0;
  ResetState;
  ResetStats;
end;

destructor TLIFNetwork.Destroy;
begin
  inherited Destroy;
end;

function TLIFNetwork.ParamCount: Integer;
begin
  Result := (FInputSize * FHiddenSize) + (FHiddenSize * FOutputSize);
end;

procedure TLIFNetwork.ResetState;
var
  i: Integer;
begin
  for i := 0 to FHiddenSize - 1 do FVHid[i] := 0.0;
  for i := 0 to FOutputSize - 1 do FVOut[i] := 0.0;
end;

procedure TLIFNetwork.ResetStats;
begin
  FSpikeTotal := 0;
  FSampleCount := 0;
end;

procedure TLIFNetwork.ComputeClassWeights(const D: TDataset);
var
  i, c0, c1: Integer;
  total: Double;
begin
  c0 := 0;
  c1 := 0;
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

function SurrogateGradFastSigmoid(V, Alpha: Double): Double; inline;
begin
  Result := Alpha / Sqr(1.0 + Alpha * Abs(V));
end;

function TLIFNetwork.ForwardSample(const Sample: TSample): TDoubleArray;
var
  t, i, j, sampleT, sampleInSize: Integer;
  currentBeta, vTh, curIn, curOut, spikeVal: Double;
begin
  SetLength(Result, FOutputSize);
  for i := 0 to FOutputSize - 1 do Result[i] := 0.0;
  ResetState;

  sampleT := Length(Sample.Data);
  if sampleT = 0 then Exit;

  currentBeta := FConfig.Beta;
  vTh := FConfig.VThreshold;

  for t := 0 to sampleT - 1 do
  begin
    sampleInSize := Length(Sample.Data[t]);
    if sampleInSize > FInputSize then sampleInSize := FInputSize;

    // Скрытый слой
    for j := 0 to FHiddenSize - 1 do
    begin
      curIn := 0.0;
      for i := 0 to sampleInSize - 1 do
        curIn := curIn + Sample.Data[t][i] * FWHid[i * FHiddenSize + j];

      FVHid[j] := currentBeta * FVHid[j] + curIn;
      if FVHid[j] >= vTh then
      begin
        spikeVal := 1.0;
        FVHid[j] := FVHid[j] - vTh;
        Inc(FSpikeTotal);
      end
      else
        spikeVal := 0.0;

      FDSpikeHid[j] := spikeVal;
    end;

    // Выходной слой
    for j := 0 to FOutputSize - 1 do
    begin
      curOut := 0.0;
      for i := 0 to FHiddenSize - 1 do
        curOut := curOut + FDSpikeHid[i] * FWOut[i * FOutputSize + j];

      FVOut[j] := currentBeta * FVOut[j] + curOut;
      Result[j] := Result[j] + FVOut[j];
    end;
  end;

  for i := 0 to FOutputSize - 1 do
    Result[i] := Result[i] / sampleT;
  Inc(FSampleCount);
end;

function TLIFNetwork.TrainSample(const Sample: TSample; LR: Double): Double;
var
  t, i, j, sampleT, sampleInSize: Integer;
  loss, mx, sum, w, classW: Double;
  currentBeta, vTh, curIn, curOut, spikeVal: Double;
  g, gnorm, scale, mhat, vhat: Double;
  b1, b2, eps, wd, clip: Double;
begin
  ResetState;
  sampleT := Length(Sample.Data);
  if sampleT = 0 then Exit(0.0);

  // Автоматическое расширение буферов времени при длинных сэмплах
  if sampleT > FAllocatedT then
  begin
    SetLength(FSpikeHidHist, sampleT);
    SetLength(FVHidHist, sampleT);
    for t := FAllocatedT to sampleT - 1 do
    begin
      SetLength(FSpikeHidHist[t], FHiddenSize);
      SetLength(FVHidHist[t], FHiddenSize);
    end;
    FAllocatedT := sampleT;
  end;

  for i := 0 to FOutputSize - 1 do FLogits[i] := 0.0;

  currentBeta := FConfig.Beta;
  vTh := FConfig.VThreshold;

  // 1. Прямой проход с фиксацией траекторий
  for t := 0 to sampleT - 1 do
  begin
    sampleInSize := Length(Sample.Data[t]);
    if sampleInSize > FInputSize then sampleInSize := FInputSize;

    for j := 0 to FHiddenSize - 1 do
    begin
      curIn := 0.0;
      for i := 0 to sampleInSize - 1 do
        curIn := curIn + Sample.Data[t][i] * FWHid[i * FHiddenSize + j];

      FVHid[j] := currentBeta * FVHid[j] + curIn;
      FVHidHist[t][j] := FVHid[j];
      if FVHid[j] >= vTh then
      begin
        spikeVal := 1.0;
        FVHid[j] := FVHid[j] - vTh;
      end
      else
        spikeVal := 0.0;

      FSpikeHidHist[t][j] := spikeVal;
    end;

    for j := 0 to FOutputSize - 1 do
    begin
      curOut := 0.0;
      for i := 0 to FHiddenSize - 1 do
        curOut := curOut + FSpikeHidHist[t][i] * FWOut[i * FOutputSize + j];

      FVOut[j] := currentBeta * FVOut[j] + curOut;
      FLogits[j] := FLogits[j] + FVOut[j];
    end;
  end;

  for i := 0 to FOutputSize - 1 do
    FLogits[i] := FLogits[i] / sampleT;

  // Softmax
  mx := FLogits[0];
  for i := 1 to FOutputSize - 1 do
    if FLogits[i] > mx then mx := FLogits[i];

  sum := 0.0;
  for i := 0 to FOutputSize - 1 do
  begin
    FProbs[i] := Exp(EnsureRange(FLogits[i] - mx, -40.0, 40.0));
    sum := sum + FProbs[i];
  end;
  for i := 0 to FOutputSize - 1 do
    FProbs[i] := FProbs[i] / Max(sum, 1e-12);

  w := 1.0;
  if (Sample.Label_ >= 0) and (Sample.Label_ <= 1) then
    w := FClassWeights[Sample.Label_];
  loss := -w * Ln(Max(FProbs[Sample.Label_], 1e-12));

  classW := w;
  for i := 0 to FOutputSize - 1 do
    FDLogits[i] := classW * FProbs[i] / sampleT;
  if (Sample.Label_ >= 0) and (Sample.Label_ < FOutputSize) then
    FDLogits[Sample.Label_] := FDLogits[Sample.Label_] - classW / sampleT;

  // Обнуление накопителей градиента
  for i := 0 to High(FGHid) do FGHid[i] := 0.0;
  for i := 0 to High(FGOut) do FGOut[i] := 0.0;

  // 2. Обратный проход во времени (BPTT)
  for t := sampleT - 1 downto 0 do
  begin
    sampleInSize := Length(Sample.Data[t]);
    if sampleInSize > FInputSize then sampleInSize := FInputSize;

    for i := 0 to FHiddenSize - 1 do
    begin
      FDSpikeHid[i] := 0.0;
      for j := 0 to FOutputSize - 1 do
      begin
        FGOut[i * FOutputSize + j] := FGOut[i * FOutputSize + j] +
          FSpikeHidHist[t][i] * FDLogits[j];
        FDSpikeHid[i] := FDSpikeHid[i] + FDLogits[j] * FWOut[i * FOutputSize + j];
      end;
      FDPreHid[i] := FDSpikeHid[i] * SurrogateGradFastSigmoid(FVHidHist[t][i], 2.0);
    end;

    for i := 0 to sampleInSize - 1 do
      for j := 0 to FHiddenSize - 1 do
        FGHid[i * FHiddenSize + j] := FGHid[i * FHiddenSize + j] +
          Sample.Data[t][i] * FDPreHid[j];
  end;

  // 3. Встроенный шаг оптимизатора Adam для FWHid и FWOut
  Inc(FTAdam);
  b1 := FConfig.Beta1;
  b2 := FConfig.Beta2;
  eps := FConfig.AdamEps;
  wd := FConfig.WeightDecay;
  clip := FConfig.GradientClip;

  // Обновление весов скрытого слоя
  gnorm := 0.0;
  for i := 0 to High(FGHid) do gnorm := gnorm + Sqr(FGHid[i]);
  gnorm := Sqrt(gnorm);
  if (clip > 0.0) and (gnorm > clip) then scale := clip / (gnorm + 1e-12)
  else scale := 1.0;

  for i := 0 to High(FWHid) do
  begin
    g := FGHid[i] * scale;
    if wd > 0.0 then g := g + wd * FWHid[i];
    FMHid[i] := b1 * FMHid[i] + (1.0 - b1) * g;
    FVmHid[i] := b2 * FVmHid[i] + (1.0 - b2) * Sqr(g);
    mhat := FMHid[i] / (1.0 - Power(b1, FTAdam));
    vhat := FVmHid[i] / (1.0 - Power(b2, FTAdam));
    FWHid[i] := FWHid[i] - LR * mhat / (Sqrt(Max(vhat, 0.0)) + eps);
  end;

  // Обновление весов выходного слоя
  gnorm := 0.0;
  for i := 0 to High(FGOut) do gnorm := gnorm + Sqr(FGOut[i]);
  gnorm := Sqrt(gnorm);
  if (clip > 0.0) and (gnorm > clip) then scale := clip / (gnorm + 1e-12)
  else scale := 1.0;

  for i := 0 to High(FWOut) do
  begin
    g := FGOut[i] * scale;
    if wd > 0.0 then g := g + wd * FWOut[i];
    FMOut[i] := b1 * FMOut[i] + (1.0 - b1) * g;
    FVmOut[i] := b2 * FVmOut[i] + (1.0 - b2) * Sqr(g);
    mhat := FMOut[i] / (1.0 - Power(b1, FTAdam));
    vhat := FVmOut[i] / (1.0 - Power(b2, FTAdam));
    FWOut[i] := FWOut[i] - LR * mhat / (Sqrt(Max(vhat, 0.0)) + eps);
  end;

  Result := loss;
end;

function TLIFNetwork.Evaluate(const D: TDataset): Double;
var
  i, j, pred, correct: Integer;
  logits: TDoubleArray;
begin
  correct := 0;
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