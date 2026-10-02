unit fsnn.experiments;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math,
  fsnn.types, fsnn.rng, fsnn.matrix, fsnn.datasets,
  fsnn.neuron, fsnn.network, fsnn.lif, fsnn.train, fsnn.stats,
  fsnn.log, fsnn.threads;

var
  MaxThreads: Integer = 1;

procedure RunOneSeed(const Cfq: TExperimentConfig; SeedValue: Integer;
  out Res: TRunResult);
procedure RunSeedsSequential(const Cfq: TExperimentConfig;
  const Seeds: array of Integer;
  out Results: TRunResultArray);
procedure RunAllExperiments;

// Процедуры расширенной исследовательской программы
procedure RunSurrogateSweep;
procedure RunAggregationSweep;
procedure RunDelayScalingSweep;

implementation

procedure RunOneSeed(const Cfq: TExperimentConfig; SeedValue: Integer;
  out Res: TRunResult);
var
  net: TSNNNetwork;
  trainSet, testSet: TDataset;
  localCfg: TExperimentConfig;
begin
  localCfg := Cfq;
  localCfg.Seed := SeedValue;
  InitRNG(SeedValue * 1000 + 1);
  trainSet := GenerateDataset(localCfg, localCfg.TrainSamples, SeedValue);
  testSet := GenerateDataset(localCfg, localCfg.TestSamples, SeedValue + 1000);
  net := TSNNNetwork.Create(localCfg);
  try
    TrainNetwork(net, trainSet, testSet, localCfg, Res, False, @LogLn);
  finally
    net.Free;
  end;
end;

procedure RunSeedsSequential(const Cfq: TExperimentConfig;
  const Seeds: array of Integer;
  out Results: TRunResultArray);
var
  NumThreads: Integer;
begin
  NumThreads := MaxThreads;
  if NumThreads < 1 then NumThreads := 1;
  RunSeedsParallel(Cfq, Seeds, NumThreads, Results, @LogLn);
end;

procedure RunStatisticsFine;
var
  config: TExperimentConfig;
  i: Integer;
  accVals, spikeVals, energyVals: array of Double;
  st: TStatsResult;
  collapseCount: Integer;
  seedResults: TRunResultArray;
begin
  Section(Format('Statistics over %d seeds (txor, depth=4 sub=2)', [NUM_SEEDS]));
  config := DefaultConfig;
  SetLength(accVals, NUM_SEEDS);
  SetLength(spikeVals, NUM_SEEDS);
  SetLength(energyVals, NUM_SEEDS);
  collapseCount := 0;
  RunSeedsSequential(config, SEEDS, seedResults);
  for i := 0 to NUM_SEEDS - 1 do
  begin
    accVals[i] := seedResults[i].TestAccuracy;
    spikeVals[i] := seedResults[i].MeanSpikeCount;
    energyVals[i] := seedResults[i].EnergyPJ;
    if seedResults[i].Collapsed then Inc(collapseCount);
    CsvWriteRun('stats', SEEDS[i], config, seedResults[i]);
  end;
  st := ComputeStats(accVals);
  LogFmt(' Accuracy: %.4f +/- %.4f (CI95 %.4f, min=%.4f, max=%.4f)',
    [st.Mean, st.Std, st.CI95, st.Min, st.Max]);
  st := ComputeStats(spikeVals);
  LogFmt(' Spikes: %.2f +/- %.2f', [st.Mean, st.Std]);
  st := ComputeStats(energyVals);
  LogFmt(' Energy: %.1f +/- %.1f pJ', [st.Mean, st.Std]);
  LogFmt(' Collapsed: %d / %d', [collapseCount, NUM_SEEDS]);
end;

procedure RunFineAblation;
var
  config: TExperimentConfig;
  depths: array[0..3] of Integer = (1, 2, 3, 4);
  subs: array[0..2] of Integer = (1, 2, 4);
  dl, sl, i: Integer;
  accVals, spikeVals, energyVals: array of Double;
  stAcc, stSpk, stEng: TStatsResult;
  baseAcc: array of Double;
  tStat, pVal: Double;
  sig: Boolean;
  first: Boolean;
  collapseCount: Integer;
  seedResults: TRunResultArray;
begin
  Section(Format('Fine ablation (%d seeds each, t-test vs depth=1 sub=1)', [NUM_SEEDS]));
  LogLn('Depth | Sub | Acc (mean+/-std) | CI95 | Collapse | Spikes | Energy | t-test');
  LogLn('------------------------------------------------------------------------------');
  SetLength(baseAcc, NUM_SEEDS);
  first := True;
  for dl := 0 to High(depths) do
    for sl := 0 to High(subs) do
    begin
      config := DefaultConfig;
      config.FractalDepth := depths[dl];
      config.SubNeuronsPerLvl := subs[sl];
      config.TrainSamples := 300;
      config.TestSamples := 100;
      config.Epochs := 40;
      config.EarlyStopPatience := 15;
      SetLength(accVals, NUM_SEEDS);
      SetLength(spikeVals, NUM_SEEDS);
      SetLength(energyVals, NUM_SEEDS);
      collapseCount := 0;
      RunSeedsSequential(config, SEEDS, seedResults);
      for i := 0 to NUM_SEEDS - 1 do
      begin
        accVals[i] := seedResults[i].TestAccuracy;
        spikeVals[i] := seedResults[i].MeanSpikeCount;
        energyVals[i] := seedResults[i].EnergyPJ;
        if seedResults[i].Collapsed then Inc(collapseCount);
        CsvWriteRun('ablation', SEEDS[i], config, seedResults[i]);
      end;
      stAcc := ComputeStats(accVals);
      stSpk := ComputeStats(spikeVals);
      stEng := ComputeStats(energyVals);
      if first then
      begin
        for i := 0 to NUM_SEEDS - 1 do baseAcc[i] := accVals[i];
        first := False;
        LogFmt('%5d | %3d | %6.3f+/-%5.3f | %5.3f | %3d/%3d | %6.1f | %6.0f | baseline',
          [depths[dl], subs[sl], stAcc.Mean, stAcc.Std, stAcc.CI95,
           collapseCount, NUM_SEEDS, stSpk.Mean, stEng.Mean]);
      end
      else
      begin
        sig := TTest(accVals, baseAcc, tStat, pVal);
        LogFmt('%5d | %3d | %6.3f+/-%5.3f | %5.3f | %3d/%3d | %6.1f | %6.0f | p=%.4f %s',
          [depths[dl], subs[sl], stAcc.Mean, stAcc.Std, stAcc.CI95,
           collapseCount, NUM_SEEDS, stSpk.Mean, stEng.Mean,
           pVal, BoolToStr(sig, 'SIG', 'ns')]);
      end;
    end;
end;

procedure RunBaselineComparison;
var
  cfgFrac4, cfgFrac6, cfgLIF: TExperimentConfig;
  frac4Res, frac6Res: TRunResultArray;
  accF4, accL4, accF6, accL6: array of Double;
  stF4, stL4, stF6, stL6: TStatsResult;
  lifNet: TLIFNetwork;
  trainSet, testSet: TDataset;
  i, epoch, s, p4, p6: Integer;
  lrNow, tStat, pVal: Double;
  sig4, sig6: Boolean;
begin
  Section('Baseline Comparison: Fractal vs Monolithic LIF (15 seeds)');
  LogLn('Architecture       | Params | Acc (mean+/-std) | CI95  | Spikes | Energy');
  LogLn('-------------------+--------+------------------+-------+--------+-------');

  cfgFrac4 := DefaultConfig;
  cfgFrac4.FractalDepth := 4;
  cfgFrac4.SubNeuronsPerLvl := 2;
  cfgFrac4.TrainSamples := 300;
  cfgFrac4.TestSamples := 100;
  cfgFrac4.Epochs := 40;
  RunSeedsSequential(cfgFrac4, SEEDS, frac4Res);
  p4 := frac4Res[0].TotalParams;

  cfgFrac6 := DefaultConfig;
  cfgFrac6.FractalDepth := 6;
  cfgFrac6.SubNeuronsPerLvl := 2;
  cfgFrac6.TrainSamples := 300;
  cfgFrac6.TestSamples := 100;
  cfgFrac6.Epochs := 40;
  RunSeedsSequential(cfgFrac6, SEEDS, frac6Res);
  p6 := frac6Res[0].TotalParams;

  SetLength(accF4, NUM_SEEDS);
  SetLength(accF6, NUM_SEEDS);
  SetLength(accL4, NUM_SEEDS);
  SetLength(accL6, NUM_SEEDS);

  for i := 0 to NUM_SEEDS - 1 do
  begin
    accF4[i] := frac4Res[i].TestAccuracy;
    accF6[i] := frac6Res[i].TestAccuracy;
  end;

  Write('  [LIF d=4 baseline] Seeds: ');
  for i := 0 to NUM_SEEDS - 1 do
  begin
    Write('.'); Flush(Output);
    cfgLIF := cfgFrac4;
    cfgLIF.Seed := SEEDS[i];
    InitRNG(SEEDS[i] * 1000 + 1);
    trainSet := GenerateDataset(cfgLIF, cfgLIF.TrainSamples, SEEDS[i]);
    testSet := GenerateDataset(cfgLIF, cfgLIF.TestSamples, SEEDS[i] + 1000);
    lifNet := TLIFNetwork.Create(cfgLIF, p4);
    try
      lifNet.ComputeClassWeights(trainSet);
      for epoch := 1 to cfgLIF.Epochs do
      begin
        lrNow := cfgLIF.LearningRate0;
        for s := 0 to High(trainSet.Samples) do
          lifNet.TrainSample(trainSet.Samples[s], lrNow);
      end;
      accL4[i] := lifNet.Evaluate(testSet);
    finally
      lifNet.Free;
    end;
  end;
  WriteLn(' done.');

  Write('  [LIF d=6 baseline] Seeds: ');
  for i := 0 to NUM_SEEDS - 1 do
  begin
    Write('.'); Flush(Output);
    cfgLIF := cfgFrac6;
    cfgLIF.Seed := SEEDS[i];
    InitRNG(SEEDS[i] * 1000 + 1);
    trainSet := GenerateDataset(cfgLIF, cfgLIF.TrainSamples, SEEDS[i]);
    testSet := GenerateDataset(cfgLIF, cfgLIF.TestSamples, SEEDS[i] + 1000);
    lifNet := TLIFNetwork.Create(cfgLIF, p6);
    try
      lifNet.ComputeClassWeights(trainSet);
      for epoch := 1 to cfgLIF.Epochs do
      begin
        lrNow := cfgLIF.LearningRate0;
        for s := 0 to High(trainSet.Samples) do
          lifNet.TrainSample(trainSet.Samples[s], lrNow);
      end;
      accL6[i] := lifNet.Evaluate(testSet);
    finally
      lifNet.Free;
    end;
  end;
  WriteLn(' done.');

  stF4 := ComputeStats(accF4);
  stL4 := ComputeStats(accL4);
  stF6 := ComputeStats(accF6);
  stL6 := ComputeStats(accL6);

  sig4 := TTest(accF4, accL4, tStat, pVal);
  LogFmt('Fractal d=4 s=2    | %6d | %6.3f +/- %5.3f | %5.3f | %6.1f | %6.0f',
    [p4, stF4.Mean, stF4.Std, stF4.CI95, frac4Res[0].MeanSpikeCount, frac4Res[0].EnergyPJ]);
  LogFmt('LIF (same params)  | %6d | %6.3f +/- %5.3f | %5.3f |   --   |   --',
    [p4, stL4.Mean, stL4.Std, stL4.CI95]);
  LogFmt('  t-test d=4 vs LIF: p=%.4f (%s)', [pVal, BoolToStr(sig4, 'SIG', 'ns')]);

  sig6 := TTest(accF6, accL6, tStat, pVal);
  LogFmt('Fractal d=6 s=2    | %6d | %6.3f +/- %5.3f | %5.3f | %6.1f | %6.0f',
    [p6, stF6.Mean, stF6.Std, stF6.CI95, frac6Res[0].MeanSpikeCount, frac6Res[0].EnergyPJ]);
  LogFmt('LIF (same params)  | %6d | %6.3f +/- %5.3f | %5.3f |   --   |   --',
    [p6, stL6.Mean, stL6.Std, stL6.CI95]);
  LogFmt('  t-test d=6 vs LIF: p=%.4f (%s)', [pVal, BoolToStr(sig6, 'SIG', 'ns')]);
end;

procedure RunDepthScalingFine;
var
  config: TExperimentConfig;
  i, dk: Integer;
  depths: array[0..6] of Integer = (1, 2, 3, 4, 5, 6, 7);
  accVals: array of Double;
  st: TStatsResult;
  paramSum, energySum: Double;
  collapseCount: Integer;
  seedResults: TRunResultArray;
begin
  Section(Format('Depth scaling at sub=2 (%d seeds)', [NUM_SEEDS]));
  LogLn('Depth | Acc (mean+/-std) | CI95 | Params | Energy | Collapse');
  LogLn('------------------------------------------------------------');
  for dk := 0 to High(depths) do
  begin
    SetLength(accVals, NUM_SEEDS);
    paramSum := 0; energySum := 0; collapseCount := 0;
    config := DefaultConfig;
    config.FractalDepth := depths[dk];
    config.SubNeuronsPerLvl := 2;
    config.TrainSamples := 300;
    config.TestSamples := 100;
    config.Epochs := 40;
    config.EarlyStopPatience := 15;
    RunSeedsSequential(config, SEEDS, seedResults);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      accVals[i] := seedResults[i].TestAccuracy;
      paramSum := paramSum + seedResults[i].TotalParams;
      energySum := energySum + seedResults[i].EnergyPJ;
      if seedResults[i].Collapsed then Inc(collapseCount);
      CsvWriteRun('depth_scaling', SEEDS[i], config, seedResults[i]);
    end;
    st := ComputeStats(accVals);
    LogFmt('%5d | %6.3f +/- %5.3f | %5.3f | %6.0f | %6.0f | %3d/%3d',
      [depths[dk], st.Mean, st.Std, st.CI95,
       paramSum / NUM_SEEDS, energySum / NUM_SEEDS,
       collapseCount, NUM_SEEDS]);
  end;
end;

procedure RunSubScalingFine;
var
  config: TExperimentConfig;
  i, sk: Integer;
  subs: array[0..5] of Integer = (1, 2, 3, 4, 6, 8);
  accVals: array of Double;
  st: TStatsResult;
  paramSum, energySum: Double;
  collapseCount: Integer;
  seedResults: TRunResultArray;
begin
  Section(Format('Sub scaling at depth=4 (%d seeds)', [NUM_SEEDS]));
  LogLn('Sub | Acc (mean+/-std) | CI95 | Params | Energy | Collapse');
  LogLn('----------------------------------------------------------');
  for sk := 0 to High(subs) do
  begin
    SetLength(accVals, NUM_SEEDS);
    paramSum := 0; energySum := 0; collapseCount := 0;
    config := DefaultConfig;
    config.FractalDepth := 4;
    config.SubNeuronsPerLvl := subs[sk];
    config.TrainSamples := 300;
    config.TestSamples := 100;
    config.Epochs := 40;
    config.EarlyStopPatience := 15;
    RunSeedsSequential(config, SEEDS, seedResults);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      accVals[i] := seedResults[i].TestAccuracy;
      paramSum := paramSum + seedResults[i].TotalParams;
      energySum := energySum + seedResults[i].EnergyPJ;
      if seedResults[i].Collapsed then Inc(collapseCount);
      CsvWriteRun('sub_scaling', SEEDS[i], config, seedResults[i]);
    end;
    st := ComputeStats(accVals);
    LogFmt('%3d | %6.3f +/- %5.3f | %5.3f | %6.0f | %6.0f | %3d/%3d',
      [subs[sk], st.Mean, st.Std, st.CI95,
       paramSum / NUM_SEEDS, energySum / NUM_SEEDS,
       collapseCount, NUM_SEEDS]);
  end;
end;

procedure RunFractalDimension;
var
  cfg: TExperimentConfig;
  net: TSNNNetwork;
  trainSet, testSet: TDataset;
  i, s, t, idx, totalSubs, seqLen, ptr: Integer;
  spikesSeries: array of Double;
  dBoxVals: array of Double;
  st: TStatsResult;
  dummyRes: TRunResult;
  activeSpikes: Integer;
  dVal: Double;
begin
  Section('Fractal Dimension of Activations (Box-Counting, 15 seeds)');
  cfg := DefaultConfig;
  cfg.FractalDepth := 4;
  cfg.SubNeuronsPerLvl := 2;
  cfg.Epochs := 20;
  SetLength(dBoxVals, NUM_SEEDS);

  for i := 0 to NUM_SEEDS - 1 do
  begin
    cfg.Seed := SEEDS[i];
    InitRNG(SEEDS[i] * 1000 + 1);
    trainSet := GenerateDataset(cfg, 120, SEEDS[i]);
    testSet := GenerateDataset(cfg, 32, SEEDS[i] + 1000);
    net := TSNNNetwork.Create(cfg);
    try
      TrainNetwork(net, trainSet, testSet, cfg, dummyRes, False, nil);

      totalSubs := net.HidFractal.OutSize * net.HidFractal.SubVal;
      seqLen := Length(testSet.Samples) * cfg.TimeSteps * totalSubs;
      SetLength(spikesSeries, seqLen);
      for idx := 0 to seqLen - 1 do spikesSeries[idx] := 0.0;

      ptr := 0;
      activeSpikes := 0;
      for s := 0 to High(testSet.Samples) do
      begin
        net.ForwardSample(testSet.Samples[s]);
        for t := 0 to cfg.TimeSteps - 1 do
          for idx := 0 to totalSubs - 1 do
          begin
            if net.HidFractal.HistSpike(t, net.HidFractal.DepthVal - 1, idx) > 0.5 then
            begin
              spikesSeries[ptr] := 1.0;
              Inc(activeSpikes);
            end;
            Inc(ptr);
          end;
      end;

      if activeSpikes > 10 then
      begin
        dVal := BoxCountingDimension(spikesSeries);
        if dVal <= 0.0 then dVal := 1.0;
        dBoxVals[i] := dVal;
      end
      else
        dBoxVals[i] := 1.0;
    finally
      net.Free;
    end;
  end;

  st := ComputeStats(dBoxVals);
  LogFmt('D_box (mean +/- std): %.4f +/- %.4f (CI95: %.4f)',
    [st.Mean, st.Std, st.CI95]);
  if st.Mean > 1.05 then
    LogLn('Hypothesis CONFIRMED: D_box > 1.0 indicates intrinsic fractal dynamics.')
  else
    LogLn('Hypothesis PARTIALLY confirmed: activity close to linear attractor (D_box ~ 1.0).');
end;

procedure RunBetaSweep;
var
  config: TExperimentConfig;
  i, k: Integer;
  betas: array[0..6] of Double = (0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999);
  accVals: array of Double;
  st: TStatsResult;
  seedResults: TRunResultArray;
begin
  Section(Format('Beta sweep on delayed match (%d seeds)', [NUM_SEEDS]));
  LogLn('Beta | Acc (mean+/-std) | CI95');
  LogLn('--------------------------------');
  for k := 0 to High(betas) do
  begin
    SetLength(accVals, NUM_SEEDS);
    config := DefaultConfig;
    config.Dataset := dkDelayedMatch;
    config.Beta := betas[k];
    config.BetaMin := betas[k];
    config.BetaMax := betas[k];
    config.TimeConstants := tcUniform;
    config.FractalDepth := 4;
    config.SubNeuronsPerLvl := 2;
    config.HiddenSize := 16;
    config.TimeSteps := 20;
    config.DelaySteps := 5;
    config.TrainSamples := 300;
    config.TestSamples := 100;
    config.Epochs := 40;
    config.EarlyStopPatience := 15;
    RunSeedsSequential(config, SEEDS, seedResults);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      accVals[i] := seedResults[i].TestAccuracy;
      CsvWriteRun('beta_sweep', SEEDS[i], config, seedResults[i]);
    end;
    st := ComputeStats(accVals);
    LogFmt('%.3f | %6.3f +/- %5.3f | %5.3f',
      [betas[k], st.Mean, st.Std, st.CI95]);
  end;
end;

procedure RunNoiseFine;
var
  config: TExperimentConfig;
  i, k: Integer;
  noises: array[0..5] of Double = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20);
  accVals: array of Double;
  st: TStatsResult;
  seedResults: TRunResultArray;
begin
  Section(Format('Noise sweep (%d seeds)', [NUM_SEEDS]));
  LogLn('Noise | Acc (mean+/-std) | CI95');
  LogLn('--------------------------------');
  for k := 0 to High(noises) do
  begin
    SetLength(accVals, NUM_SEEDS);
    config := DefaultConfig;
    config.NoiseSigma := noises[k];
    config.TrainSamples := 300;
    config.TestSamples := 100;
    config.Epochs := 40;
    config.EarlyStopPatience := 15;
    RunSeedsSequential(config, SEEDS, seedResults);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      accVals[i] := seedResults[i].TestAccuracy;
      CsvWriteRun('noise', SEEDS[i], config, seedResults[i]);
    end;
    st := ComputeStats(accVals);
    LogFmt('%.2f | %6.3f +/- %5.3f | %5.3f',
      [noises[k], st.Mean, st.Std, st.CI95]);
  end;
end;

procedure RunRobustnessFine;
var
  config: TExperimentConfig;
  i, p, q: Integer;
  pruneLevels: array[0..4] of Double = (0.1, 0.3, 0.5, 0.7, 0.9);
  quantLevels: array[0..3] of Integer = (4, 3, 2, 1);
  accVals: array of Double;
  st: TStatsResult;
  runRes: TRunResult;
  net: TSNNNetwork;
  trainSet, testSet: TDataset;
begin
  Section(Format('Prune robustness (%d seeds)', [NUM_SEEDS]));
  LogLn('Prune | Acc (mean+/-std) | CI95');
  LogLn('--------------------------------');
  for p := 0 to High(pruneLevels) do
  begin
    SetLength(accVals, NUM_SEEDS);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      config := DefaultConfig;
      config.Seed := SEEDS[i];
      config.TrainSamples := 300;
      config.TestSamples := 100;
      config.Epochs := 40;
      config.EarlyStopPatience := 15;
      InitRNG(SEEDS[i] * 1000 + 1);
      trainSet := GenerateDataset(config, config.TrainSamples, SEEDS[i]);
      testSet := GenerateDataset(config, config.TestSamples, SEEDS[i] + 1000);
      net := TSNNNetwork.Create(config);
      try
        TrainNetwork(net, trainSet, testSet, config, runRes, False, nil);
        net.PruneWeights(pruneLevels[p]);
        accVals[i] := net.Evaluate(testSet);
      finally
        net.Free;
      end;
    end;
    st := ComputeStats(accVals);
    LogFmt('%.1f | %6.3f +/- %5.3f | %5.3f',
      [pruneLevels[p], st.Mean, st.Std, st.CI95]);
  end;

  Section(Format('Quant per-channel robustness (%d seeds)', [NUM_SEEDS]));
  LogLn('Bits | Acc (mean+/-std) | CI95');
  LogLn('--------------------------------');
  for q := 0 to High(quantLevels) do
  begin
    SetLength(accVals, NUM_SEEDS);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      config := DefaultConfig;
      config.Seed := SEEDS[i];
      config.TrainSamples := 300;
      config.TestSamples := 100;
      config.Epochs := 40;
      config.EarlyStopPatience := 15;
      InitRNG(SEEDS[i] * 1000 + 1);
      trainSet := GenerateDataset(config, config.TrainSamples, SEEDS[i]);
      testSet := GenerateDataset(config, config.TestSamples, SEEDS[i] + 1000);
      net := TSNNNetwork.Create(config);
      try
        TrainNetwork(net, trainSet, testSet, config, runRes, False, nil);
        net.QuantizeWeightsPerChannel(quantLevels[q]);
        accVals[i] := net.Evaluate(testSet);
      finally
        net.Free;
      end;
    end;
    st := ComputeStats(accVals);
    LogFmt('%3d | %6.3f +/- %5.3f | %5.3f',
      [quantLevels[q], st.Mean, st.Std, st.CI95]);
  end;
end;

procedure RunKaimingGainFine;
var
  config: TExperimentConfig;
  i, k: Integer;
  gains: array[0..6] of Double = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0);
  accVals: array of Double;
  st: TStatsResult;
  collapseCount: Integer;
  seedResults: TRunResultArray;
begin
  Section(Format('Kaiming gain sweep (%d seeds)', [NUM_SEEDS]));
  LogLn('Gain | Acc (mean+/-std) | CI95 | Collapse');
  LogLn('-------------------------------------------');
  for k := 0 to High(gains) do
  begin
    SetLength(accVals, NUM_SEEDS);
    collapseCount := 0;
    config := DefaultConfig;
    config.KaimingGain := gains[k];
    config.TrainSamples := 300;
    config.TestSamples := 100;
    config.Epochs := 40;
    config.EarlyStopPatience := 15;
    RunSeedsSequential(config, SEEDS, seedResults);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      accVals[i] := seedResults[i].TestAccuracy;
      if seedResults[i].Collapsed then Inc(collapseCount);
      CsvWriteRun('kaiming', SEEDS[i], config, seedResults[i]);
    end;
    st := ComputeStats(accVals);
    LogFmt('%.2f | %6.3f +/- %5.3f | %5.3f | %3d/%3d',
      [gains[k], st.Mean, st.Std, st.CI95, collapseCount, NUM_SEEDS]);
  end;
end;

procedure RunLateralInhibitionSweep;
var
  config: TExperimentConfig;
  i, k: Integer;
  values: array[0..4] of Double = (0.0, 0.1, 0.3, 0.5, 0.7);
  accVals: array of Double;
  st: TStatsResult;
  seedResults: TRunResultArray;
begin
  Section(Format('Lateral inhibition sweep (%d seeds)', [NUM_SEEDS]));
  LogLn('Value | Acc (mean+/-std) | CI95');
  LogLn('--------------------------------');
  for k := 0 to High(values) do
  begin
    SetLength(accVals, NUM_SEEDS);
    config := DefaultConfig;
    config.LateralInhibition := values[k];
    config.TrainSamples := 300;
    config.TestSamples := 100;
    config.Epochs := 40;
    config.EarlyStopPatience := 15;
    RunSeedsSequential(config, SEEDS, seedResults);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      accVals[i] := seedResults[i].TestAccuracy;
      CsvWriteRun('lateral_inh', SEEDS[i], config, seedResults[i]);
    end;
    st := ComputeStats(accVals);
    LogFmt('%.2f | %6.3f +/- %5.3f | %5.3f',
      [values[k], st.Mean, st.Std, st.CI95]);
  end;
end;

procedure RunSurrogateSweep;
var
  config: TExperimentConfig;
  s: TSurrogateKind;
  i: Integer;
  accVals: array of Double;
  st: TStatsResult;
  seedResults: TRunResultArray;
begin
  Section(Format('Surrogate gradient functions sweep (%d seeds)', [NUM_SEEDS]));
  LogLn('Surrogate   | Acc (mean+/-std) | CI95 | Spikes | Energy');
  LogLn('------------+------------------+-------+--------+-------');
  for s := Low(TSurrogateKind) to High(TSurrogateKind) do
  begin
    SetLength(accVals, NUM_SEEDS);
    config := DefaultConfig;
    config.Surrogate := s;
    config.FractalDepth := 4;
    config.SubNeuronsPerLvl := 2;
    config.TrainSamples := 300;
    config.TestSamples := 100;
    config.Epochs := 40;
    config.EarlyStopPatience := 15;
    RunSeedsSequential(config, SEEDS, seedResults);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      accVals[i] := seedResults[i].TestAccuracy;
      CsvWriteRun('surrogate_sweep', SEEDS[i], config, seedResults[i]);
    end;
    st := ComputeStats(accVals);
    LogFmt('%-11s | %6.3f +/- %5.3f | %5.3f | %6.1f | %6.0f',
      [SurrogateName(s), st.Mean, st.Std, st.CI95,
       seedResults[0].MeanSpikeCount, seedResults[0].EnergyPJ]);
  end;
end;

procedure RunAggregationSweep;
var
  config: TExperimentConfig;
  agg: TAggregationKind;
  i: Integer;
  accVals: array of Double;
  st: TStatsResult;
  seedResults: TRunResultArray;
begin
  Section(Format('Sub-neuron aggregation benchmark (%d seeds)', [NUM_SEEDS]));
  LogLn('Aggregation | Acc (mean+/-std) | CI95 | Collapse');
  LogLn('------------+------------------+-------+---------');
  for agg := Low(TAggregationKind) to High(TAggregationKind) do
  begin
    SetLength(accVals, NUM_SEEDS);
    config := DefaultConfig;
    config.Aggregation := agg;
    config.FractalDepth := 4;
    config.SubNeuronsPerLvl := 2;
    config.TrainSamples := 300;
    config.TestSamples := 100;
    config.Epochs := 40;
    config.EarlyStopPatience := 15;
    RunSeedsSequential(config, SEEDS, seedResults);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      accVals[i] := seedResults[i].TestAccuracy;
      CsvWriteRun('aggregation_sweep', SEEDS[i], config, seedResults[i]);
    end;
    st := ComputeStats(accVals);
    LogFmt('%-11s | %6.3f +/- %5.3f | %5.3f | %3d/%3d',
      [AggregationName(agg), st.Mean, st.Std, st.CI95,
       Ord(seedResults[0].Collapsed), NUM_SEEDS]);
  end;
end;

procedure RunDelayScalingSweep;
var
  config: TExperimentConfig;
  delays: array[0..4] of Integer = (5, 10, 15, 20, 25);
  dIdx, i: Integer;
  accVals: array of Double;
  st: TStatsResult;
  seedResults: TRunResultArray;
begin
  Section(Format('Delayed match: time-lag retention scaling (%d seeds)', [NUM_SEEDS]));
  LogLn('Delay (tau) | Acc (mean+/-std) | CI95 | TimeSteps');
  LogLn('------------+------------------+-------+----------');
  for dIdx := 0 to High(delays) do
  begin
    SetLength(accVals, NUM_SEEDS);
    config := DefaultConfig;
    config.Dataset := dkDelayedMatch;
    config.DelaySteps := delays[dIdx];
    config.TimeSteps := delays[dIdx] + 10;
    config.FractalDepth := 4;
    config.SubNeuronsPerLvl := 2;
    config.TimeConstants := tcMultiScale;
    config.BetaMin := 0.85;
    config.BetaMax := 0.995;
    config.TrainSamples := 300;
    config.TestSamples := 100;
    config.Epochs := 40;
    config.EarlyStopPatience := 15;
    RunSeedsSequential(config, SEEDS, seedResults);
    for i := 0 to NUM_SEEDS - 1 do
    begin
      accVals[i] := seedResults[i].TestAccuracy;
      CsvWriteRun('delay_scaling', SEEDS[i], config, seedResults[i]);
    end;
    st := ComputeStats(accVals);
    LogFmt('%11d | %6.3f +/- %5.3f | %5.3f | %9d',
      [delays[dIdx], st.Mean, st.Std, st.CI95, config.TimeSteps]);
  end;
end;

procedure RunAllExperiments;
begin
  LogLn('########## ЧАСТЬ 2: ИССЛЕДОВАТЕЛЬСКАЯ ПРОГРАММА ##########');
  LogLn('');
  RunStatisticsFine;
  RunFineAblation;
  RunBaselineComparison;
  RunDepthScalingFine;
  RunSubScalingFine;
  RunFractalDimension;
  RunBetaSweep;
  RunNoiseFine;
  RunRobustnessFine;
  RunKaimingGainFine;
  RunLateralInhibitionSweep;
  // Новые исследовательские блоки
  RunSurrogateSweep;
  RunAggregationSweep;
  RunDelayScalingSweep;
end;

end.