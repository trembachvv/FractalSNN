program run_benchmarks;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  SysUtils, DateUtils,
  fsnn.types, fsnn.log, fsnn.experiments;

var
  startTime: TDateTime;
  OutDir, LogName, CsvName: string;
begin
  OutDir := 'results';
  if not DirectoryExists(OutDir) then
    CreateDir(OutDir);

  LogName := OutDir + DirectorySeparator + FormatDateTime('yyyymmdd_hhnnss', Now) + '_benchmarks.log';
  CsvName := OutDir + DirectorySeparator + FormatDateTime('yyyymmdd_hhnnss', Now) + '_benchmarks.csv';

  OpenLog(LogName);
  CsvOpen(CsvName);
  CsvWriteHeader;

  MaxThreads := 8; // Многопоточный пул вычислений
  startTime := Now;

  LogLn('======================================================================');
  LogLn('        FractalSNN: Полный научный бенчмарк (15 сидов)                ');
  LogLn('======================================================================');

  RunAllExperiments;

  LogLn('');
  LogFmt('Время выполнения: %.1f сек.', [(Now - startTime) * SecsPerDay]);
  CsvClose;
  CloseLog;
end.