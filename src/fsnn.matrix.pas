unit fsnn.matrix;

{$mode objfpc}{$H+}

interface

uses SysUtils, Math, fsnn.types, fsnn.rng;

type
  TMatrix = record
    Rows, Cols: Integer;
    Data: array of Double;
  end;

  TAdamState = record
    M: TMatrix;
    V: TMatrix;
    T: Int64;
  end;

function MatCreate(Rows, Cols: Integer; InitVal: Double = 0.0): TMatrix;
function MatGet(const M: TMatrix; r, c: Integer): Double;
procedure MatSet(var M: TMatrix; r, c: Integer; v: Double);
function MatMul(const A, B: TMatrix): TMatrix;
function MatSum(const M: TMatrix): Double;
procedure MatKaiming(var M: TMatrix; Gain: Double);
procedure MatPrune(var M: TMatrix; Fraction: Double);
procedure MatQuantize(var M: TMatrix; Bits: Integer);
procedure MatQuantizePerChannel(var M: TMatrix; Bits: Integer);
function AdamInit(const M: TMatrix): TAdamState;
procedure AdamStep(var Param: TMatrix; const Grad: TMatrix; var St: TAdamState;
  LR, Beta1, Beta2, Eps, WeightDecay, ClipNorm: Double);

implementation

function MatCreate(Rows, Cols: Integer; InitVal: Double = 0.0): TMatrix;
var k: Integer;
begin
  Result.Rows := Rows;
  Result.Cols := Cols;
  SetLength(Result.Data, Rows * Cols);
  for k := 0 to High(Result.Data) do Result.Data[k] := InitVal;
end;

function MatGet(const M: TMatrix; r, c: Integer): Double;
begin
  Result := M.Data[r * M.Cols + c];
end;

procedure MatSet(var M: TMatrix; r, c: Integer; v: Double);
begin
  M.Data[r * M.Cols + c] := v;
end;

function MatMul(const A, B: TMatrix): TMatrix;
var i, j, k: Integer; s: Double;
begin
  if A.Cols <> B.Rows then
    raise Exception.Create('MatMul dim mismatch');
  Result := MatCreate(A.Rows, B.Cols, 0.0);
  for i := 0 to A.Rows - 1 do
    for k := 0 to A.Cols - 1 do
    begin
      s := A.Data[i * A.Cols + k];
      if Abs(s) < EPS then Continue;
      for j := 0 to B.Cols - 1 do
        Result.Data[i * B.Cols + j] :=
          Result.Data[i * B.Cols + j] + s * B.Data[k * B.Cols + j];
    end;
end;

function MatSum(const M: TMatrix): Double;
var i: Integer;
begin
  Result := 0;
  for i := 0 to High(M.Data) do Result := Result + M.Data[i];
end;

procedure MatKaiming(var M: TMatrix; Gain: Double);
var i: Integer; scale: Double;
begin
  scale := Gain * Sqrt(2.0 / Max(1, M.Rows));
  for i := 0 to High(M.Data) do
    M.Data[i] := RandGaussian(scale);
end;

procedure MatPrune(var M: TMatrix; Fraction: Double);
var
  i, n, k, j: Integer;
  vals: array of Double;
  threshold, tmp: Double;
begin
  if Fraction <= 0 then Exit;
  n := Length(M.Data);
  SetLength(vals, n);
  for i := 0 to n - 1 do vals[i] := Abs(M.Data[i]);
  for i := 0 to n - 2 do
    for j := 0 to n - 2 - i do
      if vals[j] > vals[j + 1] then
      begin
        tmp := vals[j]; vals[j] := vals[j + 1]; vals[j + 1] := tmp;
      end;
  k := Trunc(n * Fraction);
  if k <= 0 then Exit;
  threshold := vals[k - 1];
  for i := 0 to n - 1 do
    if Abs(M.Data[i]) <= threshold then M.Data[i] := 0.0;
end;

procedure MatQuantize(var M: TMatrix; Bits: Integer);
var i: Integer; levels, vmin, vmax, scale: Double;
begin
  if Bits <= 0 then Exit;
  vmin := M.Data[0]; vmax := M.Data[0];
  for i := 1 to High(M.Data) do
  begin
    if M.Data[i] < vmin then vmin := M.Data[i];
    if M.Data[i] > vmax then vmax := M.Data[i];
  end;
  if vmax - vmin < EPS then Exit;
  levels := Power(2, Bits) - 1;
  scale := (vmax - vmin) / levels;
  for i := 0 to High(M.Data) do
    M.Data[i] := vmin + Round((M.Data[i] - vmin) / scale) * scale;
end;

procedure MatQuantizePerChannel(var M: TMatrix; Bits: Integer);
var
  i, j: Integer;
  levels, vmin, vmax, scale, v: Double;
begin
  if Bits <= 0 then Exit;
  levels := Power(2, Bits) - 1;
  if M.Rows > M.Cols then
  begin
    for i := 0 to M.Rows - 1 do
    begin
      vmin := M.Data[i * M.Cols];
      vmax := M.Data[i * M.Cols];
      for j := 1 to M.Cols - 1 do
      begin
        v := M.Data[i * M.Cols + j];
        if v < vmin then vmin := v;
        if v > vmax then vmax := v;
      end;
      if vmax - vmin < EPS then Continue;
      scale := (vmax - vmin) / levels;
      for j := 0 to M.Cols - 1 do
      begin
        v := M.Data[i * M.Cols + j];
        M.Data[i * M.Cols + j] := vmin + Round((v - vmin) / scale) * scale;
      end;
    end;
  end
  else
  begin
    for j := 0 to M.Cols - 1 do
    begin
      vmin := M.Data[j]; vmax := M.Data[j];
      for i := 1 to M.Rows - 1 do
      begin
        v := M.Data[i * M.Cols + j];
        if v < vmin then vmin := v;
        if v > vmax then vmax := v;
      end;
      if vmax - vmin < EPS then Continue;
      scale := (vmax - vmin) / levels;
      for i := 0 to M.Rows - 1 do
      begin
        v := M.Data[i * M.Cols + j];
        M.Data[i * M.Cols + j] := vmin + Round((v - vmin) / scale) * scale;
      end;
    end;
  end;
end;

function AdamInit(const M: TMatrix): TAdamState;
begin
  Result.M := MatCreate(M.Rows, M.Cols, 0.0);
  Result.V := MatCreate(M.Rows, M.Cols, 0.0);
  Result.T := 0;
end;

procedure AdamStep(var Param: TMatrix; const Grad: TMatrix; var St: TAdamState;
  LR, Beta1, Beta2, Eps, WeightDecay, ClipNorm: Double);
var
  i: Integer; g, gnorm, scale, mhat, vhat: Double;
begin
  gnorm := 0.0;
  for i := 0 to High(Grad.Data) do
    gnorm := gnorm + Grad.Data[i] * Grad.Data[i];
  gnorm := Sqrt(gnorm);
  if (ClipNorm > 0) and (gnorm > ClipNorm) then
    scale := ClipNorm / (gnorm + EPS)
  else
    scale := 1.0;
  Inc(St.T);
  for i := 0 to High(Param.Data) do
  begin
    g := Grad.Data[i] * scale;
    if WeightDecay > 0 then g := g + WeightDecay * Param.Data[i];
    St.M.Data[i] := Beta1 * St.M.Data[i] + (1 - Beta1) * g;
    St.V.Data[i] := Beta2 * St.V.Data[i] + (1 - Beta2) * g * g;
    mhat := St.M.Data[i] / (1 - Power(Beta1, St.T));
    vhat := St.V.Data[i] / (1 - Power(Beta2, St.T));
    Param.Data[i] := Param.Data[i] - LR * mhat / (Sqrt(vhat) + Eps);
  end;
end;

end.
