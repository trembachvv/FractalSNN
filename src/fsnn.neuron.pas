unit fsnn.neuron;

{$mode objfpc}{$H+}

interface

uses Math, fsnn.types, fsnn.rng, fsnn.matrix;

type
  TFracStep = record
    V:        array of array of Double;
    Spike:    array of array of Double;
    HiddenIn: array of array of Double;
  end;

  TBaseNeuron = class
  protected
    FVThreshold: Double;
    FBeta:       Double;
    FAlpha:      Double;
  public
    constructor Create(AVTh, ABeta, AAlpha: Double);
    function SurrogateGrad(V: Double): Double; virtual;
    property VThreshold: Double read FVThreshold;
    property Beta: Double read FBeta;
  end;

  TFractalNeuron = class(TBaseNeuron)
  private
    FSurrogate:   TSurrogateKind;
    FDepth:       Integer;
    FSub:         Integer;
    FAgg:         TAggregationKind;
    FInputSize:   Integer;
    FOutputSize:  Integer;
    FW:           array of TMatrix;
    FWR:          array of TMatrix;
    FAttn:        array of array of Double;
    FBetaLvl:     array of Double;
    FV:           array of array of Double;
    FPrevSpike:   array of array of Double;
    FHist:        array of TFracStep;
    FHistLen:     Integer;
    FSpikeCount:  Int64;
    FActiveNrn:   Int64;
    FDropoutP:    Double;
    FTrainMode:   Boolean;
    FRecurrent:   Boolean;
    FRecScale:    Double;
    FTimeConstants: TTimeConstantsKind;
    FBetaMin, FBetaMax: Double;
    FLateralInh:  Double;
  public
    constructor Create(AVTh, ABeta: Double; ASurrogate: TSurrogateKind;
      ADepth, ASub, AIn, AOut: Integer; AAgg: TAggregationKind;
      ADropoutP: Double; AGain: Double; ARecurrent: Boolean; ARecScale: Double;
      ATimeConstants: TTimeConstantsKind; ABetaMin, ABetaMax: Double;
      ALateralInh: Double);
    procedure SetAlpha(A: Double);
    function SurrogateGrad(V: Double): Double; override;
    function ForwardStep(const Input: TMatrix): TMatrix;
    procedure ResetState;
    procedure ClearHistory;
    procedure ResetSpikeCount;
    procedure SetTrainMode(M: Boolean);
    procedure ApplyGrad(l: Integer; const Grad: TMatrix; var St: TAdamState;
      LR, B1, B2, Eps, WD, Clip: Double);
    procedure ApplyRecGrad(l: Integer; const Grad: TMatrix; var St: TAdamState;
      LR, B1, B2, Eps, WD, Clip: Double);
    procedure PruneWeights(Fraction: Double);
    procedure QuantizeWeights(Bits: Integer);
    procedure QuantizeWeightsPerChannel(Bits: Integer);
    procedure SetWeightAt(l, i: Integer; v: Double);
    procedure ZeroAllWeights;
    function ParamCount: Integer;
    function DepthVal: Integer;
    function SubVal: Integer;
    function InSize: Integer;
    function OutSize: Integer;
    function WeightRows(l: Integer): Integer;
    function WeightCols(l: Integer): Integer;
    function WeightAt(l, i: Integer): Double;
    function RecRows(l: Integer): Integer;
    function RecCols(l: Integer): Integer;
    function RecAt(l, i: Integer): Double;
    function BetaAt(l: Integer): Double;
    function HistSpike(t, l, idx: Integer): Double;
    function HistV(t, l, idx: Integer): Double;
    function HistoryLen: Integer;
    function IsRecurrent: Boolean;
    property SpikeCount: Int64 read FSpikeCount;
    property ActiveNeurons: Int64 read FActiveNrn;
    property LateralInhibition: Double read FLateralInh write FLateralInh;
  end;

implementation

constructor TBaseNeuron.Create(AVTh, ABeta, AAlpha: Double);
begin
  inherited Create;
  FVThreshold := AVTh;
  FBeta := ABeta;
  FAlpha := AAlpha;
end;

function TBaseNeuron.SurrogateGrad(V: Double): Double;
begin
  Result := FAlpha / Sqr(1.0 + FAlpha * Abs(V));
end;

constructor TFractalNeuron.Create(AVTh, ABeta: Double; ASurrogate: TSurrogateKind;
  ADepth, ASub, AIn, AOut: Integer; AAgg: TAggregationKind;
  ADropoutP: Double; AGain: Double; ARecurrent: Boolean; ARecScale: Double;
  ATimeConstants: TTimeConstantsKind; ABetaMin, ABetaMax: Double;
  ALateralInh: Double);
var l, i: Integer;
begin
  inherited Create(AVTh, ABeta, 2.0);
  FSurrogate := ASurrogate;
  FDepth := EnsureRange(ADepth, 1, MAX_DEPTH);
  FSub := Max(1, ASub);
  FInputSize := AIn;
  FOutputSize := AOut;
  FAgg := AAgg;
  FDropoutP := ADropoutP;
  FTrainMode := True;
  FRecurrent := ARecurrent;
  FRecScale := ARecScale;
  FTimeConstants := ATimeConstants;
  FBetaMin := ABetaMin;
  FBetaMax := ABetaMax;
  FLateralInh := ALateralInh;

  SetLength(FW, FDepth);
  SetLength(FWR, FDepth);
  SetLength(FAttn, FDepth);
  SetLength(FBetaLvl, FDepth);

  for l := 0 to FDepth - 1 do
  begin
    if FTimeConstants = tcMultiScale then
    begin
      if FDepth = 1 then FBetaLvl[l] := FBeta
      else FBetaLvl[l] := FBetaMin + (FBetaMax - FBetaMin) * (l + 1) / (FDepth + 1);
    end
    else
      FBetaLvl[l] := FBeta;

    if l = 0 then
      FW[l] := MatCreate(FInputSize, FOutputSize * FSub)
    else
      FW[l] := MatCreate(FOutputSize * FSub, FOutputSize * FSub);
    MatKaiming(FW[l], AGain);

    if FRecurrent then
    begin
      FWR[l] := MatCreate(FOutputSize * FSub, FOutputSize * FSub);
      MatKaiming(FWR[l], AGain * FRecScale);
    end
    else
      FWR[l] := MatCreate(0, 0);

    SetLength(FAttn[l], FSub);
    for i := 0 to FSub - 1 do FAttn[l][i] := 1.0;
  end;

  SetLength(FV, FDepth);
  SetLength(FPrevSpike, FDepth);
  for l := 0 to FDepth - 1 do
  begin
    SetLength(FV[l], FOutputSize * FSub);
    SetLength(FPrevSpike[l], FOutputSize * FSub);
  end;
  ResetState;
  ClearHistory;
  FSpikeCount := 0;
  FActiveNrn := 0;
end;

procedure TFractalNeuron.SetAlpha(A: Double);
begin FAlpha := A; end;

procedure TFractalNeuron.SetTrainMode(M: Boolean);
begin FTrainMode := M; end;

function TFractalNeuron.SurrogateGrad(V: Double): Double;
begin
  case FSurrogate of
    skFastSigmoid: Result := FAlpha / Sqr(1.0 + FAlpha * Abs(V));
    skArctan:      Result := 1.0 / (Pi * (1.0 + Sqr(Pi * V)));
    skPiecewise:   begin
                     if Abs(V) < 1.0 then Result := FAlpha * (1 - Abs(V))
                     else Result := 0.0;
                   end;
  else
    Result := FAlpha / Sqr(1.0 + FAlpha * Abs(V));
  end;
end;

procedure TFractalNeuron.SetWeightAt(l, i: Integer; v: Double);
begin FW[l].Data[i] := v; end;

procedure TFractalNeuron.ZeroAllWeights;
var l, i: Integer;
begin
  for l := 0 to FDepth - 1 do
    for i := 0 to High(FW[l].Data) do
      FW[l].Data[i] := 0.0;
end;

function TFractalNeuron.ParamCount: Integer;
var l: Integer;
begin
  Result := 0;
  for l := 0 to FDepth - 1 do
  begin
    Inc(Result, FW[l].Rows * FW[l].Cols);
    if FRecurrent then Inc(Result, FWR[l].Rows * FWR[l].Cols);
  end;
end;

function TFractalNeuron.DepthVal: Integer; begin Result := FDepth; end;
function TFractalNeuron.SubVal: Integer; begin Result := FSub; end;
function TFractalNeuron.InSize: Integer; begin Result := FInputSize; end;
function TFractalNeuron.OutSize: Integer; begin Result := FOutputSize; end;
function TFractalNeuron.WeightRows(l: Integer): Integer; begin Result := FW[l].Rows; end;
function TFractalNeuron.WeightCols(l: Integer): Integer; begin Result := FW[l].Cols; end;
function TFractalNeuron.WeightAt(l, i: Integer): Double; begin Result := FW[l].Data[i]; end;
function TFractalNeuron.RecRows(l: Integer): Integer; begin Result := FWR[l].Rows; end;
function TFractalNeuron.RecCols(l: Integer): Integer; begin Result := FWR[l].Cols; end;
function TFractalNeuron.RecAt(l, i: Integer): Double; begin Result := FWR[l].Data[i]; end;
function TFractalNeuron.BetaAt(l: Integer): Double; begin Result := FBetaLvl[l]; end;
function TFractalNeuron.HistSpike(t, l, idx: Integer): Double; begin Result := FHist[t].Spike[l][idx]; end;
function TFractalNeuron.HistV(t, l, idx: Integer): Double; begin Result := FHist[t].V[l][idx]; end;
function TFractalNeuron.HistoryLen: Integer; begin Result := FHistLen; end;
function TFractalNeuron.IsRecurrent: Boolean; begin Result := FRecurrent; end;

procedure TFractalNeuron.PruneWeights(Fraction: Double);
var l: Integer;
begin
  for l := 0 to FDepth - 1 do
  begin
    MatPrune(FW[l], Fraction);
    if FRecurrent then MatPrune(FWR[l], Fraction);
  end;
end;

procedure TFractalNeuron.QuantizeWeights(Bits: Integer);
var l: Integer;
begin
  for l := 0 to FDepth - 1 do
  begin
    MatQuantize(FW[l], Bits);
    if FRecurrent then MatQuantize(FWR[l], Bits);
  end;
end;

procedure TFractalNeuron.QuantizeWeightsPerChannel(Bits: Integer);
var l: Integer;
begin
  for l := 0 to FDepth - 1 do
  begin
    MatQuantizePerChannel(FW[l], Bits);
    if FRecurrent then MatQuantizePerChannel(FWR[l], Bits);
  end;
end;

procedure TFractalNeuron.ResetState;
var l, i: Integer;
begin
  for l := 0 to FDepth - 1 do
  begin
    for i := 0 to High(FV[l]) do FV[l][i] := 0.0;
    for i := 0 to High(FPrevSpike[l]) do FPrevSpike[l][i] := 0.0;
  end;
end;

procedure TFractalNeuron.ClearHistory;
begin
  SetLength(FHist, 0);
  FHistLen := 0;
end;

procedure TFractalNeuron.ResetSpikeCount;
begin
  FSpikeCount := 0;
  FActiveNrn := 0;
end;

procedure TFractalNeuron.ApplyGrad(l: Integer; const Grad: TMatrix; var St: TAdamState;
  LR, B1, B2, Eps, WD, Clip: Double);
begin
  AdamStep(FW[l], Grad, St, LR, B1, B2, Eps, WD, Clip);
end;

procedure TFractalNeuron.ApplyRecGrad(l: Integer; const Grad: TMatrix; var St: TAdamState;
  LR, B1, B2, Eps, WD, Clip: Double);
begin
  AdamStep(FWR[l], Grad, St, LR, B1, B2, Eps, WD, Clip);
end;

function TFractalNeuron.ForwardStep(const Input: TMatrix): TMatrix;
var
  l, s, c, idx, b, totalSub: Integer;
  curInput, curSpikes, prevSpikes, recInput, prevState: TMatrix;
  newV, sum: Double;
  activeInput: Boolean;
  attnMax, attnSum: Double;
  attnExp: array of Double;
  currentBeta: Double;
  maxVal: Double;
  sumVal: Double;
  wSum: Double;
  sumSpikesLevel: Double;
begin
  SetLength(FHist, FHistLen + 1);
  SetLength(FHist[FHistLen].V, FDepth);
  SetLength(FHist[FHistLen].Spike, FDepth);
  SetLength(FHist[FHistLen].HiddenIn, FDepth);
  for l := 0 to FDepth - 1 do
  begin
    SetLength(FHist[FHistLen].V[l], FOutputSize * FSub);
    SetLength(FHist[FHistLen].Spike[l], FOutputSize * FSub);
    SetLength(FHist[FHistLen].HiddenIn[l], FOutputSize * FSub);
  end;

  activeInput := False;
  for b := 0 to FInputSize - 1 do
    if Abs(MatGet(Input, 0, b)) > 0.1 then activeInput := True;

  totalSub := FOutputSize * FSub;

  curInput := MatMul(Input, FW[0]);
  if FRecurrent then
  begin
    prevState := MatCreate(1, totalSub, 0.0);
    for idx := 0 to totalSub - 1 do
      MatSet(prevState, 0, idx, FPrevSpike[0][idx]);
    recInput := MatMul(prevState, FWR[0]);
    for idx := 0 to totalSub - 1 do
      curInput.Data[idx] := curInput.Data[idx] + recInput.Data[idx];
  end;

  sumSpikesLevel := 0.0;
  if FLateralInh > 0.0 then
  begin
    for s := 0 to totalSub - 1 do
      sumSpikesLevel := sumSpikesLevel + FPrevSpike[0][s];
  end;

  currentBeta := FBetaLvl[0];
  prevSpikes := MatCreate(1, totalSub, 0.0);
  for idx := 0 to totalSub - 1 do
  begin
    if FLateralInh > 0.0 then
      curInput.Data[idx] := curInput.Data[idx] - FLateralInh * (sumSpikesLevel - FPrevSpike[0][idx]);

    newV := currentBeta * FV[0][idx] + curInput.Data[idx];
    FHist[FHistLen].HiddenIn[0][idx] := curInput.Data[idx];
    if newV >= FVThreshold then
    begin
      MatSet(prevSpikes, 0, idx, 1.0);
      FHist[FHistLen].Spike[0][idx] := 1.0;
      Inc(FSpikeCount);
      if activeInput then Inc(FActiveNrn);
      newV := newV - FVThreshold;
    end
    else
    begin
      MatSet(prevSpikes, 0, idx, 0.0);
      FHist[FHistLen].Spike[0][idx] := 0.0;
    end;
    FHist[FHistLen].V[0][idx] := newV;
    FV[0][idx] := newV;
  end;

  for idx := 0 to totalSub - 1 do
    FPrevSpike[0][idx] := MatGet(prevSpikes, 0, idx);
  curSpikes := prevSpikes;

  for l := 1 to FDepth - 1 do
  begin
    curInput := MatMul(curSpikes, FW[l]);
    if FRecurrent then
    begin
      prevState := MatCreate(1, totalSub, 0.0);
      for idx := 0 to totalSub - 1 do
        MatSet(prevState, 0, idx, FPrevSpike[l][idx]);
      recInput := MatMul(prevState, FWR[l]);
      for idx := 0 to totalSub - 1 do
        curInput.Data[idx] := curInput.Data[idx] + recInput.Data[idx];
    end;

    sumSpikesLevel := 0.0;
    if FLateralInh > 0.0 then
    begin
      for s := 0 to totalSub - 1 do
        sumSpikesLevel := sumSpikesLevel + FPrevSpike[l][s];
    end;

    prevSpikes := MatCreate(1, totalSub, 0.0);
    currentBeta := FBetaLvl[l];
    for idx := 0 to totalSub - 1 do
    begin
      if FLateralInh > 0.0 then
        curInput.Data[idx] := curInput.Data[idx] - FLateralInh * (sumSpikesLevel - FPrevSpike[l][idx]);

      newV := currentBeta * FV[l][idx] + curInput.Data[idx];
      FHist[FHistLen].HiddenIn[l][idx] := curInput.Data[idx];
      if newV >= FVThreshold then
      begin
        MatSet(prevSpikes, 0, idx, 1.0);
        FHist[FHistLen].Spike[l][idx] := 1.0;
        Inc(FSpikeCount);
        newV := newV - FVThreshold;
      end
      else
      begin
        MatSet(prevSpikes, 0, idx, 0.0);
        FHist[FHistLen].Spike[l][idx] := 0.0;
      end;
      FHist[FHistLen].V[l][idx] := newV;
      FV[l][idx] := newV;
    end;
    for idx := 0 to totalSub - 1 do
      FPrevSpike[l][idx] := MatGet(prevSpikes, 0, idx);
    curSpikes := prevSpikes;
  end;

  // Изолированный потокобезопасный вызов генератора для Dropout
  if FTrainMode and (FDropoutP > 0) then
    for idx := 0 to totalSub - 1 do
      if RngUniform(ThreadRng) < FDropoutP then MatSet(curSpikes, 0, idx, 0.0);

  Result := MatCreate(1, FOutputSize, 0.0);
  for c := 0 to FOutputSize - 1 do
  begin
    case FAgg of
      agAttention:
        begin
          SetLength(attnExp, FSub);
          attnMax := -1e30;
          for s := 0 to FSub - 1 do
            if FAttn[FDepth - 1][s] > attnMax then attnMax := FAttn[FDepth - 1][s];
          attnSum := 0.0;
          for s := 0 to FSub - 1 do
          begin
            attnExp[s] := Exp(FAttn[FDepth - 1][s] - attnMax);
            attnSum := attnSum + attnExp[s];
          end;
          sum := 0.0;
          for s := 0 to FSub - 1 do
            sum := sum + (attnExp[s] / (attnSum + EPS)) *
                         MatGet(curSpikes, 0, s * FOutputSize + c);
          MatSet(Result, 0, c, sum);
        end;
      agMax:
        begin
          maxVal := -1e30;
          for s := 0 to FSub - 1 do
            if MatGet(curSpikes, 0, s * FOutputSize + c) > maxVal then
              maxVal := MatGet(curSpikes, 0, s * FOutputSize + c);
          MatSet(Result, 0, c, maxVal);
        end;
      agLearned:
        begin
          sumVal := 0.0; wSum := 0.0;
          for s := 0 to FSub - 1 do
          begin
            sumVal := sumVal + FAttn[FDepth - 1][s] *
                                 MatGet(curSpikes, 0, s * FOutputSize + c);
            wSum := wSum + Abs(FAttn[FDepth - 1][s]);
          end;
          if wSum > EPS then MatSet(Result, 0, c, sumVal / wSum)
          else MatSet(Result, 0, c, 0.0);
        end;
      agSum:
        begin
          sum := 0.0;
          for s := 0 to FSub - 1 do
            sum := sum + MatGet(curSpikes, 0, s * FOutputSize + c);
          MatSet(Result, 0, c, sum);
        end;
    else
      begin
        sum := 0.0;
        for s := 0 to FSub - 1 do
          sum := sum + MatGet(curSpikes, 0, s * FOutputSize + c);
        MatSet(Result, 0, c, sum / FSub);
      end;
    end;
  end;

  Inc(FHistLen);
end;

end.