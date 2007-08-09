{ ****************************************************************
  Info               :  X2000 Komponents
                        Freeware

  Source File Name   :  X2000.PAS
  Author             :  Baldemaier Florian (Baldemaier.Florian@gmx.net)
  Compiler           :  Delphi 4.0 Client/Server, Service Pack 3
  Decription         :  This file register all tools from X2000

  Thanks for helping :  Andreas Windisch   (wist@gmx.de)
                        Greg Nixon         (greg@nzcs.co.nz)
                        Gregory L. Bullock (bullock@mbay.net)
                        John McKnight      (Johnmc@ncilp.com)
**************************************************************** }
unit x2000;

interface

uses
   ShellAPI, Windows, Messages, SysUtils, Graphics, Controls, Forms, Classes,
   DsgnIntf, Dialogs;


procedure Register;

{$I x2000.inc}

implementation

uses x2000co, x2000lh, x2000di, x2000rc, x2000sp, x2000la, x2000si;

procedure Register;
var
  aVersion: TOSVersionInfo;
  a: boolean;
begin
  aVersion.dwOSVersionInfoSize:= SizeOf(aVersion);
  a:= GetVersionEx(aVersion) and (aVersion.dwPLatformId = VER_PLATFORM_WIN32_NT);
  if a then
   {$DEFINE NT}
  else
   {$DEFINE WIN98};

  {$IFDEF VER120}

    {$IFDEF WIN98}
      RegisterComponents('X2000', [TSystemInfo2000X]);
      RegisterComponents('X2000', [TScrPass2000X]);

      RegisterPropertyEditor(TypeInfo(TAbout2000X), TScrPass2000X,       'ABOUT', TAbout2000X);
      RegisterPropertyEditor(TypeInfo(TAbout2000X), TSystemInfo2000X,    'ABOUT', TAbout2000X);
    {$ENDIF}

    RegisterComponents('X2000', [TFileCompress2000X]);
    RegisterComponents('X2000', [TRemoveCaption2000X]);
    RegisterComponents('X2000', [TDiskInfo2000X]);
    RegisterComponents('X2000', [TLabel2000X]);

    RegisterPropertyEditor(TypeInfo(TAbout2000X), TFileCompress2000X,  'ABOUT', TAbout2000X);
    RegisterPropertyEditor(TypeInfo(TAbout2000X), TDiskInfo2000X,      'ABOUT', TAbout2000X);
    RegisterPropertyEditor(TypeInfo(TAbout2000X), TLabel2000X,         'ABOUT', TAbout2000X);
    RegisterPropertyEditor(TypeInfo(TAbout2000X), TRemoveCaption2000X, 'ABOUT', TAbout2000X);

  {$ELSE}
    MessageDlg('Sorry no Delphi 1,2,3 support. Only Delphi 4 or higher.', mtWarning, [mbOk], 0);
  {$ENDIF}

end;

end.
