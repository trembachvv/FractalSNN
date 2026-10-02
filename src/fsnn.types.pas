unit fsnn.types;
{$mode objfpc}{$H+}

interface

uses SysUtils;

type
  TDoubleArray = array of Double;
  TNeuronKind = (nkLIF, nkIF, nkAdEx, nkFractal);
  TAggregationKind = (agSum, agMean, agAttention, agMax, agLearned);
  TSurrogateKind = (skFastSigmoid, skArctan, skPiecewise);
  TDatasetKind = (dkTemporalXOR, dkDelayedMatch, dkSpokenLike,
                  dkPatternRecog, dkSequenceMem);
  TClassBalanceKind = (cbNone, cbWeights);
  TTimeConstantsKind = (tcUniform, tcMultiScale);

  TLogProc = procedure(const S: string);

  TExperimentConfig = record
    NeuronKind: TNeuronKind;
    Aggregation: TAggregationKind;
    Surrogate: TSurrogateKind;
    Dataset: TDatasetKind;
    ClassBalance: TClassBalanceKind;
    TimeConstants: TTimeConstantsKind;
    FractalDepth: Integer;
    SubNeuronsPerLvl: Integer;
    HiddenSize: Integer;
    InputSize: Integer;
    OutputSize: Integer;
    TimeSteps: Integer;
    Beta: Double;
    BetaMin: Double;
    BetaMax: Double;
    VThreshold: Double;
    SurrogateAlpha0: Double;
    SurrogateAlpha1: Double;
    LearningRate0: Double;
    LearningRate1: Double;
    WarmupEpochs: Integer;
    BatchAccum: Integer;
    Beta1, Beta2, AdamEps: Double;
    GradientClip: Double;
    WeightDecay: Double;
    SpikeDropout: Double;
    LateralInhibition: Double;
    KaimingGain: Double;
    Recurrent: Boolean;
    RecurrentScale: Double;
    Epochs: Integer;
    EarlyStopPatience: Integer;
    Seed: Integer;
    ThreadOffset: Integer;
    TrainSamples: Integer;
    TestSamples: Integer;
    NoiseSigma: Double;
    DelaySteps: Integer;
    PruneFraction: Double;
    QuantBits: Integer;
    AdaptiveThreshold: Boolean;
    SpikeRateTarget: Double;
  end;

  TRunResult = record
    TestAccuracy: Double;
    TestF1: Double;
    TestPrecision: Double;
    TestRecall: Double;
    TrainAccuracy: Double;
    MeanSpikeCount: Double;
    MeanActiveNrn: Double;
    EnergyPJ: Double;
    TotalParams: Integer;
    TrainSeconds: Double;
    EpochsRun: Integer;
    FinalLoss: Double;
    Class0PredRate: Double;
    Class1PredRate: Double;
    Collapsed: Boolean;
  end;

  TRunResultArray = array of TRunResult;

  TStatsResult = record
    Mean: Double;
    Std: Double;
    CI95: Double;
    Min: Double;
    Max: Double;
    N: Integer;
  end;

const
  E_SYNAPSE_PJ = 0.9;
  E_MEMBRANE_PJ = 4.5;
  EPS = 1e-12;
  MAX_DEPTH = 8;
  COLLAPSE_THRESHOLD = 0.90;
  NUM_SEEDS = 15;
  SEEDS: array[0..14] of Integer =
    (1, 7, 13, 42, 99, 123, 256, 512, 777, 1000,
     1234, 2024, 3141, 5555, 9999);

function DefaultConfig: TExperimentConfig;
function DatasetKindName(D: TDatasetKind): string;
function AggregationName(A: TAggregationKind): string;
function SurrogateName(S: TSurrogateKind): string;

implementation

function DefaultConfig: TExperimentConfig;
begin
  Result.NeuronKind := nkFractal;
  Result.Aggregation := agMean;
  Result.Surrogate := skFastSigmoid;
  Result.Dataset := dkTemporalXOR;
  Result.ClassBalance := cbWeights;
  Result.TimeConstants := tcMultiScale;
  Result.FractalDepth := 4;
  Result.SubNeuronsPerLvl := 2;
  Result.HiddenSize := 12;
  Result.InputSize := 4;
  Result.OutputSize := 2;
  Result.TimeSteps := 15;
  Result.Beta := 0.95;
  Result.BetaMin := 0.80;
  Result.BetaMax := 0.99;
  Result.VThreshold := 1.0;
  Result.SurrogateAlpha0 := 2.0;
  Result.SurrogateAlpha1 := 8.0;
  Result.LearningRate0 := 1e-2;
  Result.LearningRate1 := 1e-3;
  Result.WarmupEpochs := 3;
  Result.BatchAccum := 8;
  Result.Beta1 := 0.9;
  Result.Beta2 := 0.999;
  Result.AdamEps := 1e-8;
  Result.GradientClip := 0.5;
  Result.WeightDecay := 1e-4;
  Result.SpikeDropout := 0.05;
  Result.LateralInhibition := 0.0;
  Result.KaimingGain := 1.5;
  Result.Recurrent := False;
  Result.RecurrentScale := 0.3;
  Result.Epochs := 60;
  Result.EarlyStopPatience := 20;
  Result.Seed := 42;
  Result.ThreadOffset := 0;
  Result.TrainSamples := 800;
  Result.TestSamples := 300;
  Result.NoiseSigma := 0.05;
  Result.DelaySteps := 3;
  Result.PruneFraction := 0.0;
  Result.QuantBits := 0;
  Result.AdaptiveThreshold := False;
  Result.SpikeRateTarget := 0.0;
end;

function DatasetKindName(D: TDatasetKind): string;
begin
  case D of
    dkTemporalXOR: Result := 'txor';
    dkDelayedMatch: Result := 'delayed';
    dkSpokenLike: Result := 'spoken';
    dkPatternRecog: Result := 'pattern';
    dkSequenceMem: Result := 'seqmem';
    else Result := 'unknown';
  end;
end;

function AggregationName(A: TAggregationKind): string;
begin
  case A of
    agSum: Result := 'sum';
    agMean: Result := 'mean';
    agAttention: Result := 'attention';
    agMax: Result := 'max';
    agLearned: Result := 'learned';
    else Result := 'unknown';
  end;
end;

function SurrogateName(S: TSurrogateKind): string;
begin
  case S of
    skFastSigmoid: Result := 'fastsigmoid';
    skArctan: Result := 'arctan';
    skPiecewise: Result := 'piecewise';
    else Result := 'unknown';
  end;
end;

end.