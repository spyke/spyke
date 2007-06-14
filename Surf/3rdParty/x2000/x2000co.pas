{ ****************************************************************
  Info               :  TFileCompress2000X
                        Freeware

  Source File Name   :  X2000co.PAS
  Author             :  Baldemaier Florian (Baldemaier.Florian@gmx.net)
  LHA Algorithm      :  Haruhiko Okomura and Haruyasu Yoshizaki.
  LHA Modified       :  Gregory L. Bullock
  Compiler           :  Delphi 4.0 Client/Server, Service Pack 3
  Decription         :  Tool for compress a file with LHA.

  Testet on          :  Intel Pentium II, 300 Mhz and 450 Mhz, Windows 98
                        Intel Pentium III, 500 Mhz, Windows 98
                        Intel Pentium 200 Mhz, Windows98
                        D4 Service Pack 3
                        Mircosoft Windows 98
                        Mircosoft Windows 98 SE
**************************************************************** }

unit x2000co;

interface

uses
   Windows, SysUtils, Messages, Classes, Graphics, Controls, Forms,
   Dialogs, DsgnIntf, ExtCtrls, StdCtrls, ShellAPI, Filectrl, x2000lh;

type

    TAbout2000X=class(TPropertyEditor)
    public
     procedure Edit; override;
     function GetAttributes: TPropertyAttributes; override;
     function GetValue: string; override;
    end;

    TFileCompress2000X=class (TComponent)
    private
       FInput         : TFilename;
       FOutPut        : TFilename;
       FOver          : Boolean;
       FAbout         : TAbout2000X;
       procedure SetFile(value: TFilename);
       procedure SetFile2(value: TFilename);
       procedure SetAllow(value: Boolean);
    public
       procedure Compress;
       procedure Expand;
    published
       property InputFile      : TFilename   read FInput  write SetFile;
       property OutputFile     : TFilename   read FOutput write SetFile2;
       property AllowOverride  : Boolean     read FOver   write SetAllow default false;
       property About          : TAbout2000X read FAbout  write FAbout;
    end;

implementation

uses X2000About;

{$I x2000.inc}

procedure TFileCompress2000X.Expand;
var
  InStr, OutStr: TFileStream;
  FTemp: TFilename;
begin
  FInput := AnsiUppercase(FInput);
  FOutput:= AnsiUppercase(FOutput);
  if not Fileexists(FInput) then begin
     MessageDlg('Input File doen´t exists',mterror,[mbok],0);
     exit;
  end;
  if not FOver then begin
    if FInput=FOutput then begin
       MessageDlg('Input File can´t be the same as Output File',mterror,[mbok],0);
       exit;
    end;
    if FOutput='' then begin
       MessageDlg('You must enter a filename.',mterror,[mbok],0);
       exit;
    end;
    if Fileexists(FOutput) then begin
       MessageDlg('Output File does exists. You must enter a filename that are not exists.',mterror,[mbok],0);
       exit;
    end;
    InStr := TFileStream.Create(FInput,fmOpenRead);
    OutStr := TFileStream.Create(FOutput,fmCreate);
    LHAExpand(InStr, OutStr);
    InStr.Free;
    OutStr.Free;
  end;
  if FOver then begin
    FTemp:=ExtractFilepath(FInput);
    if copy(FTemp,length(FTemp),1)<>'\' then FTemp:=FTemp+'\';
    FTemp:=FTemp+'Temp.tmp';
    if FileExists(FTemp) then deleteFile(FTemp);
    InStr := TFileStream.Create(FInput,fmOpenRead);
    OutStr := TFileStream.Create(FTemp,fmCreate);
    LHAExpand(InStr, OutStr);
    try
      InStr.Free;
      OutStr.Free;
    finally
      DeleteFile (FInput);
      RenameFile (FTemp, FInPut);
    end;
  end;
end;

procedure TFileCompress2000X.Compress;
var
  InStr, OutStr: TFileStream;
  FTemp: TFilename;
begin
  FInput := AnsiUppercase(FInput);
  FOutput:= AnsiUppercase(FOutput);
  if not Fileexists(FInput) then begin
     MessageDlg('Input File doen´t exists',mterror,[mbok],0);
     exit;
  end;
  if not FOver then begin
    if Fileexists(FOutput) then begin
       MessageDlg('Output File does exists. You must enter a filename that are not exists.',mterror,[mbok],0);
       exit;
    end;
    if FOutput='' then begin
       MessageDlg('You must enter a filename.',mterror,[mbok],0);
       exit;
    end;
    if FInput=FOutput then begin
       MessageDlg('Input File can´t be the same as Output File',mterror,[mbok],0);
       exit;
    end;
    InStr  := TFileStream.Create(FInput,fmOpenRead);
    OutStr := TFileStream.Create(FOutput,fmCreate);
    LHACompress(InStr, OutStr);
    InStr.Free;
    OutStr.Free;
  end;
  if FOver then begin
    FTemp:=ExtractFilepath(FInput);
    if copy(FTemp,length(FTemp),1)<>'\' then FTemp:=FTemp+'\';
    FTemp:=FTemp+'Temp.tmp';
    if FileExists(FTemp) then deleteFile(FTemp);
    InStr := TFileStream.Create(FInput,fmOpenRead);
    OutStr := TFileStream.Create(FTemp,fmCreate);
    LHACompress(InStr, OutStr);
    try
     InStr.Free;
     OutStr.Free;
    finally
     deleteFile (FInput);
     RenameFile (FTemp, FInPut);
    end;
  end;

end;

procedure TFileCompress2000X.SetAllow(value: Boolean);
begin
  if value then begin
    FOver:=True;
    FOutput:='(not required)';
  end;
  if not value then begin
    FOver:=false;
    if FOutput='(not required)' then FOutput:='';
  end;

end;

procedure TFileCompress2000X.SetFile(value: TFilename);
begin
   if fileexists(value) then FInput:=value;
end;

procedure TFileCompress2000X.SetFile2(value: TFilename);
begin
   if FOver then FOutput:='(not required)';
   if not FOver then FOutput:=value;
end;

procedure TAbout2000X.Edit;
begin
 with TAboutForm.Create(Application) do begin
  try
    ShowModal;
  finally
    Free;
  end;
 end;
end;

function TAbout2000X.GetAttributes: TPropertyAttributes;
begin
    Result := [paMultiSelect, paDialog, paReadOnly];
end;

function TAbout2000X.GetValue: string;
begin
    Result := '(X2000)';
end;

end.
