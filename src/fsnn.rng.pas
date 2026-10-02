unit fsnn.rng;

{$mode objfpc}{$H+}
{$inline on}

interface

uses Math;

type
  TRng = record
    State: LongWord;
  end;

{ Изолированное состояние генератора для каждого потока ОС }
threadvar
  ThreadRng: TRng;

procedure InitRNG(Seed: Integer); inline;
function RandGaussian(Sigma: Double): Double; inline;
function RandIntRange(N: Integer): Integer; inline;

procedure RngSeed(var RR: TRng; S: LongWord); inline;
function RngNext(var RR: TRng): LongWord; inline;
function RngUniform(var RR: TRng): Double; inline;
function RngGaussian(var RR: TRng; Sigma: Double): Double; inline;
function RngRange(var RR: TRng; N: Integer): Integer; inline;

implementation

procedure RngSeed(var RR: TRng; S: LongWord);
begin
  RR.State := LongWord(S);
  if RR.State = 0 then RR.State := 2463534242;
end;

function RngNext(var RR: TRng): LongWord;
begin
  RR.State := RR.State xor (RR.State shl 13);
  RR.State := RR.State xor (RR.State shr 17);
  RR.State := RR.State xor (RR.State shl 5);
  Result := RR.State;
end;

function RngUniform(var RR: TRng): Double;
begin
  Result := RngNext(RR) / 4294967296.0;
end;

function RngGaussian(var RR: TRng; Sigma: Double): Double;
var u1, u2, r: Double;
begin
  repeat
    u1 := RngUniform(RR);
  until u1 > 1e-12;
  u2 := RngUniform(RR);
  r := Sqrt(-2.0 * Ln(u1));
  Result := Sigma * r * Cos(2.0 * Pi * u2);
end;

function RngRange(var RR: TRng; N: Integer): Integer;
begin
  if N <= 0 then Result := 0
  else Result := Integer(RngNext(RR) mod LongWord(N));
end;

procedure InitRNG(Seed: Integer);
begin
  RngSeed(ThreadRng, LongWord(Seed));
end;

function RandGaussian(Sigma: Double): Double;
begin
  Result := RngGaussian(ThreadRng, Sigma);
end;

function RandIntRange(N: Integer): Integer;
begin
  Result := RngRange(ThreadRng, N);
end;

end.