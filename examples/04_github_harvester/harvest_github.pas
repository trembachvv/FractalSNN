program harvest_github;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math,
  fphttpclient, opensslsockets, fpjson, jsonparser;

const
  OUT_CSV = 'github_code.csv';
  MAX_SAMPLES = 200;
  WINDOW_SIZE = 20;
  FEATURE_DIM = 8;

type
  TTokenVector = array[0..FEATURE_DIM - 1] of Double;
  TFeatureMatrix = array of TTokenVector;

  TRepoTarget = record
    Owner: string;
    Repo: string;
    LabelClass: Integer;
  end;

const
  TARGETS: array[0..2] of TRepoTarget = (
    (Owner: 'psf';           Repo: 'requests'; LabelClass: 0),
    (Owner: 'tiangolo';      Repo: 'fastapi';  LabelClass: 1),
    (Owner: 'TheAlgorithms'; Repo: 'Python';   LabelClass: 0)
  );

function HttpGet(const URL: string): string;
var
  Client: TFPHTTPClient;
begin
  Result := '';
  Client := TFPHTTPClient.Create(nil);
  try
    Client.AddHeader('User-Agent', 'FreePascal-FractalSNN-Harvester');
    Client.AddHeader('Accept', 'application/vnd.github.v3+json');
    Client.AllowRedirect := True;
    try
      Result := Client.Get(URL);
    except
      on E: Exception do
        WriteLn('  [HTTP Warning] ', E.Message, ' при запросе ', URL);
    end;
  finally
    Client.Free;
  end;
end;

function ExtractFeatures(const CodeStr: string): TFeatureMatrix;
var
  Lines: TStringList;
  i, j, tCount, indLen: Integer;
  line, trimmed: string;
  hasKW, hasHazard, hasAsync: Double;
  kwList: array[0..5] of string = ('def', 'class', 'async', 'await', 'return', 'if');
  hazardList: array[0..3] of string = ('eval', 'os.system', 'exec', 'pass');
begin
  SetLength(Result, WINDOW_SIZE);
  for i := 0 to WINDOW_SIZE - 1 do
    for j := 0 to FEATURE_DIM - 1 do
      Result[i][j] := 0.0;

  Lines := TStringList.Create;
  try
    Lines.Text := CodeStr;
    tCount := 0;

    for i := 0 to Lines.Count - 1 do
    begin
      line := Lines[i];
      trimmed := Trim(line);
      if (trimmed = '') or (Copy(trimmed, 1, 1) = '#') then Continue;

      indLen := 0;
      while (indLen < Length(line)) and (line[indLen + 1] = ' ') do
        Inc(indLen);

      hasKW := 0.0;
      for j := 0 to High(kwList) do
        if Pos(kwList[j], trimmed) > 0 then begin hasKW := 1.0; Break; end;

      hasHazard := 0.0;
      for j := 0 to High(hazardList) do
        if Pos(hazardList[j], trimmed) > 0 then begin hasHazard := 1.0; Break; end;

      hasAsync := 0.0;
      if (Pos('async', trimmed) > 0) or (Pos('await', trimmed) > 0) then
        hasAsync := 1.0;

      if Pos(':', trimmed) > 0 then Result[tCount][0] := 0.7
      else if Pos('(', trimmed) > 0 then Result[tCount][0] := 0.4
      else Result[tCount][0] := 0.1;

      Result[tCount][1] := Min(Length(trimmed), 60) / 60.0;
      Result[tCount][2] := hasKW;
      if (Pos('(', trimmed) > 0) or (Pos(')', trimmed) > 0) or (Pos('->', trimmed) > 0) then
        Result[tCount][3] := 1.0
      else
        Result[tCount][3] := 0.0;

      Result[tCount][4] := Min(indLen, 16) / 16.0;
      Result[tCount][5] := hasHazard;
      Result[tCount][6] := hasAsync;
      Result[tCount][7] := 0.5;

      Inc(tCount);
      if tCount >= WINDOW_SIZE then Break;
    end;
  finally
    Lines.Free;
  end;
end;

procedure HarvestGitHub;
var
  tIdx, i, totalCollected: Integer;
  apiUrl, jsonResp, rawCode, dUrl, fName: string;
  jData: TJSONData;
  jArr: TJSONArray;
  jObj: TJSONObject;
  feats: TFeatureMatrix;
  csvFile: TextFile;
  w, d: Integer;
  fs: TFormatSettings;
begin
  fs.DecimalSeparator := '.';
  WriteLn('======================================================================');
  WriteLn('     FractalSNN: Сборщик обучающих выборок GitHub на Free Pascal      ');
  WriteLn('======================================================================');

  AssignFile(csvFile, OUT_CSV);
  Rewrite(csvFile);

  totalCollected := 0;

  try
    for tIdx := 0 to High(TARGETS) do
    begin
      WriteLn(Format('Запрос репозитория %s/%s...', [TARGETS[tIdx].Owner, TARGETS[tIdx].Repo]));
      apiUrl := Format('https://api.github.com/repos/%s/%s/contents', 
        [TARGETS[tIdx].Owner, TARGETS[tIdx].Repo]);
      
      jsonResp := HttpGet(apiUrl);
      if jsonResp = '' then Continue;

      try
        jData := GetJSON(jsonResp);
        try
          if jData is TJSONArray then
          begin
            jArr := TJSONArray(jData);
            for i := 0 to jArr.Count - 1 do
            begin
              if totalCollected >= MAX_SAMPLES then Break;

              jObj := TJSONObject(jArr.Items[i]);
              fName := jObj.Get('name', '');

              if LowerCase(ExtractFileExt(fName)) = '.py' then
              begin
                dUrl := jObj.Get('download_url', '');
                if dUrl <> '' then
                begin
                  Write('  Скачивание: ', fName, ' ... ');
                  rawCode := HttpGet(dUrl);
                  
                  if Length(rawCode) > 50 then
                  begin
                    feats := ExtractFeatures(rawCode);
                    
                    Write(csvFile, TARGETS[tIdx].LabelClass);
                    for w := 0 to WINDOW_SIZE - 1 do
                      for d := 0 to FEATURE_DIM - 1 do
                        Write(csvFile, ',', FloatToStrF(feats[w][d], ffFixed, 0, 4, fs));
                    WriteLn(csvFile);

                    Inc(totalCollected);
                    WriteLn('OK (образцы: ', totalCollected, '/', MAX_SAMPLES, ')');
                  end
                  else
                    WriteLn('пропущен (короткий)');
                  
                  Sleep(200); // Ограничение частоты обращений к API
                end;
              end;
            end;
          end;
        finally
          jData.Free;
        end;
      except
        on E: Exception do
          WriteLn('  [JSON Error] ', E.Message);
      end;

      if totalCollected >= MAX_SAMPLES then Break;
    end;
  finally
    CloseFile(csvFile);
  end;

  WriteLn('----------------------------------------------------------------------');
  WriteLn(Format('Сбор завершен. Собрано сэмплов: %d. Файл: %s', [totalCollected, OUT_CSV]));
  WriteLn('----------------------------------------------------------------------');
end;

begin
  HarvestGitHub;
end.