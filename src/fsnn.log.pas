unit fsnn.log;

{$mode objfpc}{$H+}

interface

uses SysUtils, fsnn.types;

procedure OpenLog(const FileName: string);
procedure CloseLog;
procedure LogLn(const S: string);
procedure LogFmt(const Fmt: string; const Args: array of const);
procedure LogSection(const Name: string);
procedure Section(const Name: string);

procedure CsvOpen(const FileName: string);
procedure CsvClose;
procedure CsvWriteHeader;
procedure CsvWriteRun(const ExpName: string; Seed: Integer;
  const Cfg: TExperimentConfig; const Res: TRunResult);
function IsCsvOpened: Boolean;

implementation

var
  LogFile: TextFile;
  LogOpened: Boolean = False;
  CsvFile: TextFile;
  CsvOpened: Boolean = False;

procedure OpenLog(const FileName: string);
begin
  AssignFile(LogFile, FileName);
  {$I-} Rewrite(LogFile); {$I+}
  LogOpened := (IOResult = 0);
end;

procedure CloseLog;
begin
  if LogOpened then
  begin
    CloseFile(LogFile);
    LogOpened := False;
  end;
end;

procedure LogLn(const S: string);
begin
  WriteLn(S);
  if LogOpened then
  begin
    WriteLn(LogFile, S);
    Flush(LogFile);
  end;
end;

procedure LogFmt(const Fmt: string; const Args: array of const);
begin
  LogLn(Format(Fmt, Args));
end;

procedure LogSection(const Name: string);
begin
  LogLn('');
  LogLn('=== ' + Name + ' ===');
end;

procedure Section(const Name: string);
begin
  LogSection(Name);
end;

procedure CsvOpen(const FileName: string);
begin
  AssignFile(CsvFile, FileName);
  {$I-} Rewrite(CsvFile); {$I+}
  CsvOpened := (IOResult = 0);
end;

procedure CsvClose;
begin
  if CsvOpened then
  begin
    CloseFile(CsvFile);
    CsvOpened := False;
  end;
end;

procedure CsvWriteHeader;
begin
  if not CsvOpened then Exit;
  WriteLn(CsvFile,
    'experiment,seed,dataset,depth,sub,beta_min,beta_max,' +
    'aggregation,surrogate,recurrent,accuracy,f1,spikes,energy,params,collapsed');
  Flush(CsvFile);
end;

procedure CsvWriteRun(const ExpName: string; Seed: Integer;
  const Cfg: TExperimentConfig; const Res: TRunResult);
begin
  if not CsvOpened then Exit;
  WriteLn(CsvFile, Format('%s,%d,%s,%d,%d,%.2f,%.2f,%s,%s,%d,%.4f,%.4f,%.2f,%.1f,%d,%d',
    [ExpName, Seed, DatasetKindName(Cfg.Dataset),
     Cfg.FractalDepth, Cfg.SubNeuronsPerLvl,
     Cfg.BetaMin, Cfg.BetaMax,
     AggregationName(Cfg.Aggregation), SurrogateName(Cfg.Surrogate),
     Ord(Cfg.Recurrent),
     Res.TestAccuracy, Res.TestF1, Res.MeanSpikeCount,
     Res.EnergyPJ, Res.TotalParams, Ord(Res.Collapsed)]));
  Flush(CsvFile);
end;

function IsCsvOpened: Boolean;
begin
  Result := CsvOpened;
end;

end.