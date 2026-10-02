program test_suite;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  SysUtils, DateUtils,
  fsnn.log, fsnn.tests;

var
  allPassed: Boolean;
  logFileName, csvFileName: string;
begin
  WriteLn('======================================================================');
  WriteLn('          FractalSNN: Automated Continuous Integration Suite          ');
  WriteLn('======================================================================');

  // Формируем понятные имена: YYYYMMDD_HHNNSS_tests.log и .csv
  logFileName := FormatDateTime('yyyymmdd_hhnnss', Now) + '_tests.log';
  csvFileName := FormatDateTime('yyyymmdd_hhnnss', Now) + '_tests.csv';

  OpenLog(logFileName);
  CsvOpen(csvFileName);
  CsvWriteHeader;
  try
    allPassed := RunAllTests;
  finally
    CsvClose;
    CloseLog;
    // Временный CSV после завершения тестов можно удалить, а информативный лог оставить
    if FileExists(csvFileName) then DeleteFile(csvFileName);
  end;

  if allPassed then
  begin
    WriteLn('CI VERDICT: ALL TESTS PASSED SUCCESSFULLY (100% PASS).');
    WriteLn('Детальный протокол сохранен в: ' + logFileName);
    Halt(0);
  end
  else
  begin
    WriteLn('CI VERDICT: FAILURES DETECTED.');
    Halt(1);
  end;
end.