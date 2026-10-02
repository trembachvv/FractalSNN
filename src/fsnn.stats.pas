unit fsnn.stats;

{$mode objfpc}{$H+}

interface

uses SysUtils, Math, fsnn.types;

function ComputeStats(const Vals: array of Double): TStatsResult;
function TTest(const A, B: array of Double; out TStat, PValue: Double): Boolean;
function ErfApprox(X: Double): Double;
function BoxCountingDimension(const Series: array of Double): Double;

implementation

function ErfApprox(X: Double): Double;
const
  A1 =  0.254829592;
  A2 = -0.284496736;
  A3 =  1.421413741;
  A4 = -1.453152027;
  A5 =  1.061405429;
  P  =  0.3275911;
var
  Sign: Integer;
  T, Y: Double;
begin
  Sign := 1;
  if X < 0 then Sign := -1;
  X := Abs(X);
  T := 1.0 / (1.0 + P * X);
  Y := 1.0 - (((((A5 * T + A4) * T) + A3) * T + A2) * T + A1) * T * Exp(-X * X);
  Result := Sign * Y;
end;

function ComputeStats(const Vals: array of Double): TStatsResult;
var
  i, n: Integer;
  sum, mean, varSum, diff: Double;
begin
  n := Length(Vals);
  Result.N := n;
  if n = 0 then
  begin
    Result.Mean := 0; Result.Std := 0; Result.CI95 := 0;
    Result.Min := 0; Result.Max := 0;
    Exit;
  end;
  sum := 0;
  for i := 0 to n - 1 do sum := sum + Vals[i];
  mean := sum / n;
  varSum := 0;
  for i := 0 to n - 1 do
  begin
    diff := Vals[i] - mean;
    varSum := varSum + diff * diff;
  end;
  Result.Mean := mean;
  Result.Std := Sqrt(varSum / n);
  Result.CI95 := 1.96 * Result.Std / Sqrt(n);
  Result.Min := Vals[0];
  Result.Max := Vals[0];
  for i := 1 to n - 1 do
  begin
    if Vals[i] < Result.Min then Result.Min := Vals[i];
    if Vals[i] > Result.Max then Result.Max := Vals[i];
  end;
end;

function TTest(const A, B: array of Double; out TStat, PValue: Double): Boolean;
var
  i, nA, nB: Integer;
  mA, mB, vA, vB, se, diff, t: Double;
begin
  nA := Length(A);
  nB := Length(B);
  if (nA < 2) or (nB < 2) then
  begin
    TStat := 0; PValue := 1; Exit(False);
  end;
  mA := 0; mB := 0;
  for i := 0 to nA - 1 do mA := mA + A[i];
  for i := 0 to nB - 1 do mB := mB + B[i];
  mA := mA / nA;
  mB := mB / nB;
  vA := 0; vB := 0;
  for i := 0 to nA - 1 do vA := vA + Sqr(A[i] - mA);
  for i := 0 to nB - 1 do vB := vB + Sqr(B[i] - mB);
  vA := vA / (nA - 1);
  vB := vB / (nB - 1);
  se := Sqrt(vA / nA + vB / nB);
  if se < EPS then
  begin
    TStat := 0; PValue := 1; Exit(False);
  end;
  diff := mA - mB;
  t := diff / se;
  TStat := t;
  PValue := 2 * (1 - 0.5 * (1 + ErfApprox(Abs(t) / Sqrt(2))));
  Result := PValue < 0.05;
end;

function BoxCountingDimension(const Series: array of Double): Double;
var
  Epsilons: array[0..3] of Integer = (2, 4, 8, 16);
  LogN, LogInvEps: array[0..3] of Double;
  i, k, start, count, nonEmpty, stopIdx: Integer;
  n: Integer;
  sumX, sumY, sumXY, sumXX, denom: Double;
begin
  n := Length(Series);
  if n < 16 then
  begin
    Result := 0.0;
    Exit;
  end;

  for k := 0 to 3 do
  begin
    count := 0;
    start := 0;
    while start < n do
    begin
      nonEmpty := 0;
      stopIdx := start + Epsilons[k] - 1;
      if stopIdx > n - 1 then stopIdx := n - 1;
      for i := start to stopIdx do
        if Abs(Series[i]) > 0.5 then Inc(nonEmpty);
      if nonEmpty > 0 then Inc(count);
      start := start + Epsilons[k];
    end;
    if count < 1 then count := 1;
    LogN[k] := Ln(Double(count));
    LogInvEps[k] := -Ln(Double(Epsilons[k]));
  end;

  sumX := 0; sumY := 0; sumXY := 0; sumXX := 0;
  for k := 0 to 3 do
  begin
    sumX := sumX + LogInvEps[k];
    sumY := sumY + LogN[k];
    sumXY := sumXY + LogInvEps[k] * LogN[k];
    sumXX := sumXX + LogInvEps[k] * LogInvEps[k];
  end;
  denom := 4.0 * sumXX - sumX * sumX;
  if Abs(denom) < EPS then
  begin
    Result := 0.0;
    Exit;
  end;
  Result := (4.0 * sumXY - sumX * sumY) / denom;
end;
end.