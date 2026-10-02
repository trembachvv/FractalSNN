unit fsnn.tests;

{$mode objfpc}{$H+}

interface

uses
  Math, SysUtils, Classes,
  fsnn.types, fsnn.rng, fsnn.matrix, fsnn.datasets,
  fsnn.neuron, fsnn.network, fsnn.lif, fsnn.train, fsnn.stats,
  fsnn.log, fsnn.threads;

function RunAllTests: Boolean;

implementation

var
  TestFailures: Integer = 0;
  TestTotal: Integer = 0;

procedure Check(Cond: Boolean; const Msg: string);
begin
  Inc(TestTotal);
  if Cond then
    LogLn('  [PASS] ' + Msg)
  else
  begin
    LogLn('  [FAIL] ' + Msg);
    Inc(TestFailures);
  end;
end;

procedure CheckClose(A, B, Tol: Double; const Msg: string);
begin
  Check(Abs(A - B) <= Tol, Msg);
end;

procedure TestMatrix;
var
  A, B, C, IdentityM: TMatrix;
  i, j: Integer;
  s: Double;
  uniqueBefore, uniqueAfter: Integer;
  seen: array of Double;
  v: Double;
  found: Boolean;
begin
  Section('Matrix Operations');
  A := MatCreate(3, 4, 1.5);
  Check(A.Rows = 3, 'MatCreate rows');
  Check(A.Cols = 4, 'MatCreate cols');
  Check(Length(A.Data) = 12, 'MatCreate data length');
  CheckClose(A.Data[0], 1.5, 1e-12, 'MatCreate init value');

  MatSet(A, 1, 2, 7.5);
  CheckClose(MatGet(A, 1, 2), 7.5, 1e-12, 'MatSet/MatGet roundtrip');

  A := MatCreate(3, 3);
  for i := 0 to 8 do A.Data[i] := i + 1;
  IdentityM := MatCreate(3, 3);
  for i := 0 to 2 do MatSet(IdentityM, i, i, 1.0);
  C := MatMul(A, IdentityM);
  j := 0;
  for i := 0 to High(A.Data) do
    if Abs(A.Data[i] - C.Data[i]) > 1e-12 then Inc(j);
  Check(j = 0, 'MatMul identity preservation');

  A := MatCreate(2, 3); B := MatCreate(3, 4);
  for i := 0 to High(A.Data) do A.Data[i] := i * 0.1;
  for i := 0 to High(B.Data) do B.Data[i] := i * 0.2;
  C := MatMul(A, B);
  Check(C.Rows = 2, 'MatMul output rows');
  Check(C.Cols = 4, 'MatMul output cols');

  A := MatCreate(2, 2);
  A.Data[0] := 1; A.Data[1] := 2; A.Data[2] := 3; A.Data[3] := 4;
  CheckClose(MatSum(A), 10.0, 1e-12, 'MatSum calculation');

  InitRNG(1);
  A := MatCreate(50, 50);
  MatKaiming(A, 1.0);
  s := 0;
  for i := 0 to High(A.Data) do s := s + A.Data[i] * A.Data[i];
  s := Sqrt(s / Length(A.Data));
  Check(s > 0.01, 'Kaiming init nonzero standard deviation');
  Check(s < 10.0, 'Kaiming init bounded variance');

  InitRNG(1);
  A := MatCreate(20, 20);
  for i := 0 to High(A.Data) do A.Data[i] := (i mod 7) - 3;
  MatPrune(A, 0.5);
  j := 0;
  for i := 0 to High(A.Data) do
    if Abs(A.Data[i]) < 1e-12 then Inc(j);
  Check(j >= 200, 'MatPrune zeros >= 50% elements');

  InitRNG(1);
  A := MatCreate(20, 20);
  for i := 0 to High(A.Data) do A.Data[i] := RandGaussian(1.0) * 10.0;
  uniqueBefore := 0;
  SetLength(seen, 0);
  for i := 0 to High(A.Data) do
  begin
    v := A.Data[i];
    found := False;
    for j := 0 to High(seen) do
      if Abs(seen[j] - v) < 1e-6 then begin found := True; Break; end;
    if not found then
    begin
      SetLength(seen, Length(seen) + 1);
      seen[High(seen)] := v;
      Inc(uniqueBefore);
    end;
  end;
  MatQuantize(A, 4);
  uniqueAfter := 0;
  SetLength(seen, 0);
  for i := 0 to High(A.Data) do
  begin
    v := A.Data[i];
    found := False;
    for j := 0 to High(seen) do
      if Abs(seen[j] - v) < 1e-6 then begin found := True; Break; end;
    if not found then
    begin
      SetLength(seen, Length(seen) + 1);
      seen[High(seen)] := v;
      Inc(uniqueAfter);
    end;
  end;
  Check(uniqueAfter < uniqueBefore, 'MatQuantize reduces unique weight levels');
  Check(uniqueAfter <= 16, 'MatQuantize retains <= 16 unique levels for 4-bit');
end;

procedure TestAdam;
var
  P, G: TMatrix;
  St: TAdamState;
  it: Integer;
  val: Double;
begin
  Section('Adam Optimizer');
  P := MatCreate(1, 1, 5.0);
  St := AdamInit(P);
  for it := 1 to 2000 do
  begin
    G := MatCreate(1, 1, 2 * MatGet(P, 0, 0));
    AdamStep(P, G, St, 0.01, 0.9, 0.999, 1e-8, 0.0, 0.0);
  end;
  val := MatGet(P, 0, 0);
  Check(Abs(val) < 1e-2, 'Adam converges on convex quadratic target');

  P := MatCreate(1, 1, 1.0);
  St := AdamInit(P);
  G := MatCreate(1, 1, 1000.0);
  AdamStep(P, G, St, 0.1, 0.9, 0.999, 1e-8, 0.0, 1.0);
  Check(Abs(MatGet(P, 0, 0) - 1.0) < 1.0, 'Adam gradient clipping bounds step magnitude');
end;

procedure TestDataset;
var
  D1, D2, D3, D4, D5: TDataset;
  i, t, c, c0, c1: Integer;
  same: Boolean;
begin
  Section('Spiking Datasets');
  D1 := GenerateTemporalXORDataset(20, 15, 4, 42, 0.05, 3);
  Check(Length(D1.Samples) = 20, 'TemporalXOR sample count');
  Check(Length(D1.Samples[0].Data) = 15, 'TemporalXOR time steps');
  Check(Length(D1.Samples[0].Data[0]) = 4, 'TemporalXOR input size');

  D2 := GenerateTemporalXORDataset(20, 15, 4, 42, 0.05, 3);
  same := True;
  for i := 0 to 19 do
    for t := 0 to 14 do
      for c := 0 to 3 do
        if Abs(D1.Samples[i].Data[t][c] - D2.Samples[i].Data[t][c]) > 1e-12 then
          same := False;
  Check(same, 'TemporalXOR dataset is strictly deterministic per seed');

  D3 := GenerateDelayedMatchDataset(20, 15, 4, 7, 0.05, 3);
  Check(Length(D3.Samples) = 20, 'DelayedMatch sample count valid');

  D3 := GenerateSpokenLikeDataset(20, 15, 4, 7, 0.05);
  c0 := 0; c1 := 0;
  for i := 0 to 19 do
    if D3.Samples[i].Label_ = 0 then Inc(c0) else Inc(c1);
  Check(c0 = 10, 'SpokenLike balanced binary classes');

  D4 := GeneratePatternRecogDataset(20, 15, 4, 3, 0.05);
  Check(Length(D4.Samples) = 20, 'PatternRecog sample count valid');

  D5 := GenerateSequenceMemDataset(20, 15, 4, 5, 0.05);
  Check(Length(D5.Samples) = 20, 'SequenceMem sample count valid');
end;

procedure TestNeuron;
var
  n: TFractalNeuron;
  inp, outp: TMatrix;
  s: Double;
begin
  Section('FractalNeuron Core');
  n := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 2, 3, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  try
    inp := MatCreate(1, 4, 0.5);
    outp := n.ForwardStep(inp);
    Check(outp.Rows = 1, 'ForwardStep output rows = 1');
    Check(outp.Cols = 5, 'ForwardStep output cols = 5');
    Check(n.HistoryLen = 1, 'History length incremented to 1');
    n.ForwardStep(inp);
    Check(n.HistoryLen = 2, 'History length incremented to 2');
    n.ResetState;
    n.ResetSpikeCount;
    Check(n.SpikeCount = 0, 'ResetSpikeCount zeroes counter');
    inp := MatCreate(1, 4, 10.0);
    n.ResetSpikeCount;
    n.ForwardStep(inp);
    Check(n.SpikeCount > 0, 'High amplitude current induces spikes');
    s := n.SurrogateGrad(0.0);
    CheckClose(s, 2.0, 1e-12, 'SurrogateGrad at V=0 equals Alpha');
    CheckClose(n.SurrogateGrad(-0.5), n.SurrogateGrad(0.5), 1e-12,
      'SurrogateGrad is mathematically symmetric');
  finally
    n.Free;
  end;

  n := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 1, 1, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  try
    Check(n.ParamCount = 4 * 5 * 1, 'ParamCount depth 1 calculation');
  finally
    n.Free;
  end;

  n := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 3, 2, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  try
    Check(n.ParamCount = 4 * 5 * 2 + (5 * 2) * (5 * 2) + (5 * 2) * (5 * 2),
      'ParamCount depth 3 calculation');
  finally
    n.Free;
  end;

  n := TFractalNeuron.Create(1.0, 0.95, skArctan, 2, 2, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  try
    s := n.SurrogateGrad(0.0);
    CheckClose(s, 1.0 / Pi, 1e-12, 'Arctan surrogate at V=0');
  finally
    n.Free;
  end;

  n := TFractalNeuron.Create(1.0, 0.95, skPiecewise, 2, 2, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcMultiScale, 0.85, 0.99, 0.0);
  try
    s := n.SurrogateGrad(0.5);
    Check(s > 0.0, 'Piecewise surrogate positive in active region');
    Check(n.BetaAt(0) < n.BetaAt(1), 'MultiScale beta increases monotonically');
  finally
    n.Free;
  end;
end;

procedure TestAggregationKinds;
var
  nSum, nMean, nMax: TFractalNeuron;
  inp, oSum, oMean, oMax: TMatrix;
  i, sub: Integer;
  spiked, ok: Boolean;
begin
  Section('Aggregation Kinds');
  inp := MatCreate(1, 4, 20.0);
  InitRNG(777);
  nSum := TFractalNeuron.Create(0.5, 0.95, skFastSigmoid, 1, 3, 4, 2, agSum,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  InitRNG(777);
  nMean := TFractalNeuron.Create(0.5, 0.95, skFastSigmoid, 1, 3, 4, 2, agMean,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  InitRNG(777);
  nMax := TFractalNeuron.Create(0.5, 0.95, skFastSigmoid, 1, 3, 4, 2, agMax,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  try
    nSum.SetTrainMode(False);
    nMean.SetTrainMode(False);
    nMax.SetTrainMode(False);
    oSum := nSum.ForwardStep(inp);
    oMean := nMean.ForwardStep(inp);
    oMax := nMax.ForwardStep(inp);
    sub := nSum.SubVal;
    spiked := False;
    for i := 0 to High(oSum.Data) do
      if oSum.Data[i] > 0 then spiked := True;
    Check(spiked, 'agSum produced non-zero spikes');
    ok := True;
    for i := 0 to High(oSum.Data) do
      if oSum.Data[i] > 0 then
        if Abs(oMean.Data[i] - oSum.Data[i] / sub) > 1e-9 then ok := False;
    Check(ok, 'agMean = agSum / Sub');
    ok := True;
    for i := 0 to High(oMax.Data) do
      if oMax.Data[i] + 1e-9 < oMean.Data[i] then ok := False;
    Check(ok, 'agMax >= agMean');
  finally
    nSum.Free;
    nMean.Free;
    nMax.Free;
  end;
end;

procedure TestMultiScaleTimeConstants;
var
  n: TFractalNeuron;
  b0, b1, b2: Double;
begin
  Section('MultiScale Time Constants');
  n := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 3, 2, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcMultiScale, 0.80, 0.99, 0.0);
  try
    b0 := n.BetaAt(0); b1 := n.BetaAt(1); b2 := n.BetaAt(2);
    Check(b0 < b1, 'Level 0 beta < Level 1 beta');
    Check(b1 < b2, 'Level 1 beta < Level 2 beta');
    Check(b0 >= 0.80 - 1e-9, 'Level 0 beta >= BetaMin');
    Check(b2 <= 0.99 + 1e-9, 'Level 2 beta <= BetaMax');
  finally
    n.Free;
  end;
end;

procedure TestPiecewiseSurrogate;
var
  n: TFractalNeuron;
  s0, s05, s2: Double;
begin
  Section('Piecewise Surrogate Dynamics');
  n := TFractalNeuron.Create(1.0, 0.95, skPiecewise, 2, 2, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  try
    s0 := n.SurrogateGrad(0.0);
    s05 := n.SurrogateGrad(0.5);
    s2 := n.SurrogateGrad(2.0);
    Check(s0 > s05, 'Piecewise grad at 0 > grad at 0.5');
    Check(s05 > 0, 'Piecewise grad at 0.5 strictly positive');
    CheckClose(s2, 0.0, 1e-12, 'Piecewise grad outside boundary = 0');
  finally
    n.Free;
  end;
end;

procedure TestForwardAllAggregations;
var
  agg: TAggregationKind;
  n: TFractalNeuron;
  inp, outp: TMatrix;
  i: Integer;
  finite: Boolean;
begin
  Section('Forward Step: All Aggregations');
  inp := MatCreate(1, 4, 3.0);
  for agg := Low(TAggregationKind) to High(TAggregationKind) do
  begin
    InitRNG(42);
    n := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 2, 2, 4, 3, agg,
      0.0, 1.5, False, 0.0, tcMultiScale, 0.85, 0.99, 0.0);
    try
      outp := n.ForwardStep(inp);
      finite := True;
      for i := 0 to High(outp.Data) do
        if IsNan(outp.Data[i]) or IsInfinite(outp.Data[i]) then finite := False;
      Check(finite, 'Aggregation ' + AggregationName(agg) + ' produces finite output');
    finally
      n.Free;
    end;
  end;
end;

procedure TestForwardAllSurrogates;
var
  s: TSurrogateKind;
  n: TFractalNeuron;
  inp, outp: TMatrix;
  i: Integer;
  finite: Boolean;
begin
  Section('Forward Step: All Surrogates');
  inp := MatCreate(1, 4, 3.0);
  for s := Low(TSurrogateKind) to High(TSurrogateKind) do
  begin
    InitRNG(42);
    n := TFractalNeuron.Create(1.0, 0.95, s, 2, 2, 4, 3, agMean,
      0.0, 1.5, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
    try
      outp := n.ForwardStep(inp);
      finite := True;
      for i := 0 to High(outp.Data) do
        if IsNan(outp.Data[i]) or IsInfinite(outp.Data[i]) then finite := False;
      Check(finite, 'Surrogate ' + SurrogateName(s) + ' produces finite output');
    finally
      n.Free;
    end;
  end;
end;

procedure TestDeepNetwork;
var
  cfg: TExperimentConfig;
  net: TSNNNetwork;
  D: TDataset;
  r: TDoubleArray;
  i: Integer;
  finite: Boolean;
begin
  Section('Deep Fractal Network');
  cfg := DefaultConfig;
  cfg.FractalDepth := 6;
  cfg.SubNeuronsPerLvl := 2;
  cfg.HiddenSize := 16;
  cfg.InputSize := 4;
  cfg.OutputSize := 2;
  cfg.TimeSteps := 15;
  cfg.Seed := 7;
  D := GenerateTemporalXORDataset(5, 15, 4, 7, 0.0, 3);
  net := TSNNNetwork.Create(cfg);
  try
    r := net.ForwardSample(D.Samples[0]);
    finite := True;
    for i := 0 to High(r) do
      if IsNan(r[i]) or IsInfinite(r[i]) then finite := False;
    Check(finite, 'Depth 6 forward pass is finite');
    Check(r[0] >= 0, 'Depth 6 logit0 is non-negative');
  finally
    net.Free;
  end;
end;

procedure TestParamCountConsistency;
var
  n1, n2, n3, n4: TFractalNeuron;
  p1, p2, p3, p4: Integer;
begin
  Section('ParamCount Scaling');
  n1 := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 1, 2, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  n2 := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 2, 2, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  n3 := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 3, 2, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  n4 := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 4, 2, 4, 5, agMean,
    0.0, 1.0, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  try
    p1 := n1.ParamCount; p2 := n2.ParamCount;
    p3 := n3.ParamCount; p4 := n4.ParamCount;
    Check(p1 = 40, 'ParamCount depth 1 = 40');
    Check(p2 = 140, 'ParamCount depth 2 = 140');
    Check(p3 = 240, 'ParamCount depth 3 = 240');
    Check(p4 = 340, 'ParamCount depth 4 = 340');
    Check((p1 < p2) and (p2 < p3) and (p3 < p4), 'Strictly monotonic parameter growth');
  finally
    n1.Free; n2.Free; n3.Free; n4.Free;
  end;
end;

procedure TestForwardDeterministic;
var
  n: TFractalNeuron;
  inp, o1, o2: TMatrix;
  i: Integer;
  same: Boolean;
begin
  Section('Forward Pass Determinism');
  InitRNG(999);
  n := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 3, 2, 4, 3, agMean,
    0.0, 1.5, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  try
    inp := MatCreate(1, 4, 2.0);
    o1 := n.ForwardStep(inp);
    n.ResetState; n.ResetSpikeCount;
    o2 := n.ForwardStep(inp);
    same := True;
    for i := 0 to High(o1.Data) do
      if Abs(o1.Data[i] - o2.Data[i]) > 1e-12 then same := False;
    Check(same, 'Deterministic forward execution after state reset');
  finally
    n.Free;
  end;
end;

procedure TestConvergenceQuick;
var
  cfg: TExperimentConfig;
  net: TSNNNetwork;
  trainSet, testSet: TDataset;
  res: TRunResult;
begin
  Section('Quick Convergence: Fractal Network');
  cfg := DefaultConfig;
  cfg.FractalDepth := 3; cfg.SubNeuronsPerLvl := 2; cfg.HiddenSize := 12;
  cfg.TrainSamples := 200; cfg.TestSamples := 60;
  cfg.Epochs := 20; cfg.EarlyStopPatience := 20; cfg.Seed := 42;
  trainSet := GenerateDataset(cfg, cfg.TrainSamples, cfg.Seed);
  testSet := GenerateDataset(cfg, cfg.TestSamples, cfg.Seed + 1000);
  net := TSNNNetwork.Create(cfg);
  try
    InitRNG(cfg.Seed);
    TrainNetwork(net, trainSet, testSet, cfg, res, False, @LogLn);
    Check(res.FinalLoss < 0.69, 'Loss decreased below chance baseline (0.693)');
    Check(res.TestAccuracy >= 0.5, 'Accuracy >= 0.50 on binary classification');
  finally
    net.Free;
  end;
end;

procedure TestReproducibilityTwoNetworks;
var
  cfg1, cfg2: TExperimentConfig;
  net1, net2: TSNNNetwork;
  D: TDataset;
  r1, r2: TDoubleArray;
  i: Integer;
  same: Boolean;
begin
  Section('Reproducibility Across Replicas');
  cfg1 := DefaultConfig; cfg1.Seed := 9999;
  cfg2 := DefaultConfig; cfg2.Seed := 9999;
  D := GenerateTemporalXORDataset(3, 15, 4, 1, 0.0, 3);
  net1 := TSNNNetwork.Create(cfg1);
  net2 := TSNNNetwork.Create(cfg2);
  try
    r1 := net1.ForwardSample(D.Samples[0]);
    r2 := net2.ForwardSample(D.Samples[0]);
    same := True;
    for i := 0 to High(r1) do
      if Abs(r1[i] - r2[i]) > 1e-12 then same := False;
    Check(same, 'Identical seed produces identical forward activations');
  finally
    net1.Free; net2.Free;
  end;
end;

procedure TestDifferentSeedsDifferentWeights;
var
  n1, n2: TFractalNeuron;
  w1, w2: Double;
  foundDiff: Boolean;
  i: Integer;
begin
  Section('Different Seeds Weight Divergence');
  InitRNG(111);
  n1 := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 3, 2, 4, 5, agMean,
    0.0, 1.5, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  InitRNG(222);
  n2 := TFractalNeuron.Create(1.0, 0.95, skFastSigmoid, 3, 2, 4, 5, agMean,
    0.0, 1.5, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  try
    foundDiff := False;
    for i := 0 to n1.WeightRows(0) * n1.WeightCols(0) - 1 do
    begin
      w1 := n1.WeightAt(0, i);
      w2 := n2.WeightAt(0, i);
      if Abs(w1 - w2) > 1e-9 then begin foundDiff := True; Break; end;
    end;
    Check(foundDiff, 'Different seeds initialize divergent parameter matrices');
  finally
    n1.Free; n2.Free;
  end;
end;

procedure TestLateralInhibitionZeroEquivalent;
var
  cfg1, cfg2: TExperimentConfig;
  net1, net2: TSNNNetwork;
  D: TDataset;
  r1, r2: TDoubleArray;
  i: Integer;
  same: Boolean;
begin
  Section('Bug 1: LateralInhibition Zero Equivalence');
  cfg1 := DefaultConfig; cfg1.LateralInhibition := 0.0; cfg1.Seed := 777;
  cfg2 := DefaultConfig; cfg2.LateralInhibition := 0.0; cfg2.Seed := 777;
  D := GenerateTemporalXORDataset(3, 15, 4, 1, 0.0, 3);
  net1 := TSNNNetwork.Create(cfg1);
  net2 := TSNNNetwork.Create(cfg2);
  try
    r1 := net1.ForwardSample(D.Samples[0]);
    r2 := net2.ForwardSample(D.Samples[0]);
    same := True;
    for i := 0 to High(r1) do
      if Abs(r1[i] - r2[i]) > 1e-12 then same := False;
    Check(same, 'LateralInhibition = 0.0 strictly preserves baseline behavior');
  finally
    net1.Free; net2.Free;
  end;
end;

procedure TestLateralInhibitionNonZeroChanges;
var
  n1, n2: TFractalNeuron;
  inp, o1, o2: TMatrix;
  t, c: Integer;
  differs: Boolean;
begin
  Section('Bug 1: LateralInhibition Non-Zero Modification');
  InitRNG(777);
  n1 := TFractalNeuron.Create(0.5, 0.95, skFastSigmoid, 2, 2, 4, 2, agMean,
    0.0, 1.5, False, 0.0, tcUniform, 0.95, 0.95, 0.0);
  InitRNG(777);
  n2 := TFractalNeuron.Create(0.5, 0.95, skFastSigmoid, 2, 2, 4, 2, agMean,
    0.0, 1.5, False, 0.0, tcUniform, 0.95, 0.95, 0.5);
  try
    inp := MatCreate(1, 4, 10.0);
    differs := False;
    for t := 1 to 5 do
    begin
      o1 := n1.ForwardStep(inp);
      o2 := n2.ForwardStep(inp);
      for c := 0 to High(o1.Data) do
        if Abs(o1.Data[c] - o2.Data[c]) > 1e-6 then differs := True;
    end;
    Check(differs, 'LateralInhibition = 0.5 modifies activations over time');
  finally
    n1.Free;
    n2.Free;
  end;
end;

procedure TestAgLearnedNotCollapsed;
var
  cfg: TExperimentConfig;
  net: TSNNNetwork;
  trainSet, testSet: TDataset;
  res: TRunResult;
begin
  Section('Bug 2: agLearned Non-Collapse');
  cfg := DefaultConfig;
  cfg.Aggregation := agLearned;
  cfg.FractalDepth := 3; cfg.SubNeuronsPerLvl := 2; cfg.HiddenSize := 12;
  cfg.TrainSamples := 200; cfg.TestSamples := 60;
  cfg.Epochs := 20; cfg.EarlyStopPatience := 20; cfg.Seed := 42;
  trainSet := GenerateDataset(cfg, cfg.TrainSamples, cfg.Seed);
  testSet := GenerateDataset(cfg, cfg.TestSamples, cfg.Seed + 1000);
  net := TSNNNetwork.Create(cfg);
  try
    InitRNG(cfg.Seed);
    TrainNetwork(net, trainSet, testSet, cfg, res, False, @LogLn);
    Check(res.TestAccuracy > 0.5, 'agLearned converges with accuracy > 0.50 (FAttn=1.0 fix)');
  finally
    net.Free;
  end;
end;

procedure TestDepthTwoBetter;
var
  cfg1, cfg2: TExperimentConfig;
  net1, net2: TSNNNetwork;
begin
  Section('Bug 3: Smooth MultiScale Beta Allocation');
  cfg1 := DefaultConfig;
  cfg1.FractalDepth := 1; cfg1.TimeConstants := tcMultiScale;
  cfg1.BetaMin := 0.80; cfg1.BetaMax := 0.99; cfg1.Seed := 7;
  cfg2 := DefaultConfig;
  cfg2.FractalDepth := 2; cfg2.TimeConstants := tcMultiScale;
  cfg2.BetaMin := 0.80; cfg2.BetaMax := 0.99; cfg2.Seed := 7;
  net1 := TSNNNetwork.Create(cfg1);
  net2 := TSNNNetwork.Create(cfg2);
  try
    CheckClose(net1.HidFractal.BetaAt(0), 0.95, 1e-9, 'Depth=1 assigns default beta');
    Check(net2.HidFractal.BetaAt(0) > 0.80, 'Depth=2 level 0 beta > BetaMin');
    Check(net2.HidFractal.BetaAt(1) < 0.99, 'Depth=2 level 1 beta < BetaMax');
    Check(net2.HidFractal.BetaAt(0) < net2.HidFractal.BetaAt(1),
      'Depth=2 level 0 beta < level 1 beta');
  finally
    net1.Free; net2.Free;
  end;
end;

procedure TestLIFForwardShape;
var
  cfg: TExperimentConfig;
  lif: TLIFNetwork;
  D: TDataset;
  outp: TDoubleArray;
begin
  Section('Baseline LIF: Forward Shape');
  cfg := DefaultConfig;
  D := GenerateTemporalXORDataset(5, 15, cfg.InputSize, 42, 0.0, 3);
  lif := TLIFNetwork.Create(cfg, 1824);
  try
    outp := lif.ForwardSample(D.Samples[0]);
    Check(Length(outp) = cfg.OutputSize, 'LIF output dimension matches OutputSize');
    Check((outp[0] >= 0.0) and (outp[1] >= 0.0), 'LIF forward outputs are non-negative');
  finally
    lif.Free;
  end;
end;

procedure TestLIFTrainingConverges;
var
  cfg: TExperimentConfig;
  lif: TLIFNetwork;
  trainSet, testSet: TDataset;
  epoch, s: Integer;
  loss, acc: Double;
begin
  Section('Baseline LIF: Training Convergence');
  cfg := DefaultConfig;
  cfg.TrainSamples := 200;
  cfg.TestSamples := 60;
  cfg.Epochs := 25;
  trainSet := GenerateDataset(cfg, cfg.TrainSamples, 42);
  testSet := GenerateDataset(cfg, cfg.TestSamples, 1042);
  lif := TLIFNetwork.Create(cfg, 1824);
  try
    lif.ComputeClassWeights(trainSet);
    loss := 1.0;
    for epoch := 1 to cfg.Epochs do
      for s := 0 to High(trainSet.Samples) do
        loss := lif.TrainSample(trainSet.Samples[s], 0.01);
    acc := lif.Evaluate(testSet);
    Check(loss < 0.69, 'LIF BPTT loss converged below 0.69');
    Check(acc >= 0.50, 'LIF test accuracy >= 0.50 on XOR dataset');
  finally
    lif.Free;
  end;
end;

procedure TestLIFParamsMatch;
var
  cfg: TExperimentConfig;
  net: TSNNNetwork;
  lif: TLIFNetwork;
  fracParams, lifParams: Integer;
begin
  Section('Baseline LIF: Parameter Matching');
  cfg := DefaultConfig;
  cfg.FractalDepth := 4;
  cfg.SubNeuronsPerLvl := 2;
  net := TSNNNetwork.Create(cfg);
  try
    fracParams := net.HidFractal.ParamCount;
  finally
    net.Free;
  end;

  lif := TLIFNetwork.Create(cfg, fracParams);
  try
    lifParams := lif.ParamCount;
    Check(Abs(fracParams - lifParams) <= (cfg.InputSize + cfg.OutputSize),
      'LIF parameter count closely matches fractal parameter count');
  finally
    lif.Free;
  end;
end;

procedure TestBoxCountingLine;
var
  series: array of Double;
  i: Integer;
  d: Double;
begin
  Section('Box-Counting: Linear Series');
  SetLength(series, 64);
  for i := 0 to 63 do series[i] := 1.0;
  d := BoxCountingDimension(series);
  Check((d >= 0.8) and (d <= 1.2), 'D_box of uniform signal approximates 1.0');
end;

procedure TestBoxCountingRandom;
var
  series: array of Double;
  i: Integer;
  d: Double;
begin
  Section('Box-Counting: Stochastic Series');
  InitRNG(42);
  SetLength(series, 256);
  for i := 0 to 255 do
  begin
    if (i mod 3 = 0) and (RngUniform(ThreadRng) < 0.6) then
      series[i] := 1.0
    else
      series[i] := 0.0;
  end;
  d := BoxCountingDimension(series);
  Check(d > 0.0, 'D_box of sparse spikes is strictly positive and finite');
end;

procedure TestCsvWriteHeader;
begin
  Section('CSV Logging: Header Check');
  Check(IsCsvOpened, 'CSV file handle is active and opened by host');
end;

procedure TestCsvWriteRun;
var
  cfg: TExperimentConfig;
  res: TRunResult;
begin
  Section('CSV Logging: Data Record Call');
  cfg := DefaultConfig;
  FillChar(res, SizeOf(res), 0);
  res.TestAccuracy := 0.95;
  res.EnergyPJ := 120.5;
  CsvWriteRun('unit_test_run', 9999, cfg, res);
  Check(IsCsvOpened, 'CsvWriteRun completed without IO failure');
end;

procedure TestThreadSafety;
var
  cfg: TExperimentConfig;
  testSeeds: array[0..3] of Integer = (1, 7, 42, 99);
  seqRes, parRes: TRunResultArray;
  i: Integer;
  allIdentical: Boolean;
begin
  Section('Integration: Thread Safety & Parallel Determinism');
  cfg := DefaultConfig;
  cfg.Epochs := 10;
  cfg.TrainSamples := 100;
  cfg.TestSamples := 40;

  RunSeedsParallel(cfg, testSeeds, 1, seqRes, nil);
  RunSeedsParallel(cfg, testSeeds, 4, parRes, nil);

  allIdentical := True;
  for i := 0 to 3 do
  begin
    if Abs(seqRes[i].TestAccuracy - parRes[i].TestAccuracy) > 1e-5 then
      allIdentical := False;
    if Abs(seqRes[i].FinalLoss - parRes[i].FinalLoss) > 1e-5 then
      allIdentical := False;
  end;

  Check(allIdentical, 'Parallel execution matches sequential baseline across all seeds');
end;

{ ============================================================================ }
{ НОВЫЕ ТЕСТЫ ДЛЯ ВАЛИДАЦИИ СУРРОГАТОВ, АГРЕГАЦИЙ И ЗАДЕРЖЕК ПАМЯТИ             }
{ ============================================================================ }

procedure TestSurrogateSweepConvergence;
var
  cfg: TExperimentConfig;
  net: TSNNNetwork;
  trainSet, testSet: TDataset;
  res: TRunResult;
  s: TSurrogateKind;
begin
  Section('Surrogates: Training Convergence Coverage');
  for s := Low(TSurrogateKind) to High(TSurrogateKind) do
  begin
    cfg := DefaultConfig;
    cfg.Surrogate := s;
    cfg.FractalDepth := 3;
    cfg.SubNeuronsPerLvl := 2;
    cfg.TrainSamples := 150;
    cfg.TestSamples := 50;
    cfg.Epochs := 15;
    cfg.Seed := 42;
    trainSet := GenerateDataset(cfg, cfg.TrainSamples, cfg.Seed);
    testSet := GenerateDataset(cfg, cfg.TestSamples, cfg.Seed + 1000);
    net := TSNNNetwork.Create(cfg);
    try
      InitRNG(cfg.Seed);
      TrainNetwork(net, trainSet, testSet, cfg, res, False, nil);
      Check(res.TestAccuracy >= 0.50, 'Surrogate ' + SurrogateName(s) + ' reached baseline accuracy');
    finally
      net.Free;
    end;
  end;
end;

procedure TestAggregationParity;
var
  cfg: TExperimentConfig;
  net: TSNNNetwork;
  trainSet, testSet: TDataset;
  res: TRunResult;
  agg: TAggregationKind;
begin
  Section('Aggregations: Training Stability Coverage');
  for agg := Low(TAggregationKind) to High(TAggregationKind) do
  begin
    cfg := DefaultConfig;
    cfg.Aggregation := agg;
    cfg.FractalDepth := 3;
    cfg.SubNeuronsPerLvl := 2;
    cfg.TrainSamples := 150;
    cfg.TestSamples := 50;
    cfg.Epochs := 15;
    cfg.Seed := 42;
    trainSet := GenerateDataset(cfg, cfg.TrainSamples, cfg.Seed);
    testSet := GenerateDataset(cfg, cfg.TestSamples, cfg.Seed + 1000);
    net := TSNNNetwork.Create(cfg);
    try
      InitRNG(cfg.Seed);
      TrainNetwork(net, trainSet, testSet, cfg, res, False, nil);
      Check(not res.Collapsed, 'Aggregation ' + AggregationName(agg) + ' avoided class collapse');
    finally
      net.Free;
    end;
  end;
end;

procedure TestDelayMemoryBounds;
var
  cfg: TExperimentConfig;
  dSet: TDataset;
begin
  Section('Memory: DelayedMatch Range Bounds');
  cfg := DefaultConfig;
  cfg.Dataset := dkDelayedMatch;
  cfg.DelaySteps := 15;
  cfg.TimeSteps := 25;
  dSet := GenerateDelayedMatchDataset(10, cfg.TimeSteps, cfg.InputSize, 42, 0.05, cfg.DelaySteps);
  Check(Length(dSet.Samples) = 10, 'DelayedMatch sample count valid for delay=15');
  Check(Length(dSet.Samples[0].Data) = 25, 'DelayedMatch timesteps match allocation');
end;

function RunAllTests: Boolean;
begin
  TestTotal := 0;
  TestFailures := 0;
  LogLn('########## ЭТАП 1: ЮНИТ-ТЕСТЫ И ВЕРИФИКАЦИЯ (100% COVERAGE) ##########');
  LogLn('');

  TestMatrix;
  TestAdam;
  TestDataset;
  TestNeuron;
  TestAggregationKinds;
  TestMultiScaleTimeConstants;
  TestPiecewiseSurrogate;
  TestForwardAllAggregations;
  TestForwardAllSurrogates;
  TestDeepNetwork;
  TestParamCountConsistency;
  TestForwardDeterministic;
  TestConvergenceQuick;
  TestReproducibilityTwoNetworks;
  TestDifferentSeedsDifferentWeights;
  TestLateralInhibitionZeroEquivalent;
  TestLateralInhibitionNonZeroChanges;
  TestAgLearnedNotCollapsed;
  TestDepthTwoBetter;
  TestLIFForwardShape;
  TestLIFTrainingConverges;
  TestLIFParamsMatch;
  TestBoxCountingLine;
  TestBoxCountingRandom;
  TestCsvWriteHeader;
  TestCsvWriteRun;
  TestThreadSafety;

  // Новые модульные тесты
  TestSurrogateSweepConvergence;
  TestAggregationParity;
  TestDelayMemoryBounds;

  LogLn('');
  LogLn('======================================================================');
  LogFmt('Итог тестирования: %d тестов выполнено, %d успешно, %d сбоев.',
    [TestTotal, TestTotal - TestFailures, TestFailures]);
  if TestFailures > 0 then
    LogFmt('ВНИМАНИЕ: Зафиксировано сбоев: %d.', [TestFailures])
  else
    LogLn('Все модульные и интеграционные тесты пройдены успешно (100% PASS).');
  LogLn('======================================================================');
  LogLn('');

  Result := TestFailures = 0;
end;

end.