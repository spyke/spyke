{ ****************************************************************
  Info               :  TDiskInfo2000X 
                        Freeware

  Source File Name   :  X2000Di.PAS
  Author             :  Baldemaier Florian (Baldemaier.Florian@gmx.net)
  Compiler           :  Delphi 4.0 Client/Server, Service Pack 3
  Decription         :  DiskInformation. Supports HD´s over 2 GB.

  Testet on          :  Intel Pentium II, 300 Mhz and 450 Mhz, Windows 98
                        Intel Pentium III, 500 Mhz, Windows 98
                        Intel Pentium 200 Mhz, Windows98
                        D4 Service Pack 3
                        Mircosoft Windows 98
                        Mircosoft Windows 98 SE
**************************************************************** }
unit x2000di;

interface

uses
     Windows, SysUtils, Messages, Classes, Graphics, Controls, Forms,
     Dialogs, DsgnIntf, ExtCtrls, StdCtrls, ShellAPI;

type string1 = string[1];
type string30 = string[30];

type

  TAbout2000X=class(TPropertyEditor)
  public
    procedure Edit; override;
    function GetAttributes: TPropertyAttributes; override;
    function GetValue: string; override;
  end;

  TDiskInfo2000X = class(TComponent)
  private
    FSectorsPerCluster       : DWORD;
    FBytesPerSector          : DWORD;
    FFreeClusters            : DWORD;
    FClusters                : DWORD;
    FDiskFreeSpace           : String;
    FTotalDiskSpace          : String;
    FDrive                   : string1;
    FVolumeName              : string;
    FSerialNumber            : string;
    FFileSystemType          : string;
    FAbout                   : TAbout2000X;
    function GetDriveTyp     : string30;
    procedure SetNone (value: DWord);
    procedure SetNone2(value: String);
    procedure SetNone3(value: String30);
  public
    function  IsDriveReady(DriveLetter:pChar):bool;
    constructor create(AOwner:TComponent); override;
  published
    procedure SetDrive(value : string1);
    property About             : TAbout2000X read FAbout             write FAbout;
    property Drive             : string1     read FDrive             write setdrive;
    property SectorsPerCluster : DWord       read FSectorsPerCluster write SetNone;
    property BytesPerSector    : DWord       read FBytesPerSector    write SetNone;
    property FreeClusters      : DWord       read FFreeClusters      write SetNone;
    property Clusters          : DWord       read FClusters          write SetNone;
    property DiskFreeSpace     : string      read FDiskFreeSpace     write SetNone2;
    property TotalDiskSpace    : string      read FTotalDiskSpace    write SetNone2;
    property DriveType         : string30    read GetDriveTyp        write SetNone3;
    property VolumeName        : string      read FVolumeName        write SetNone2;
    property SerialNumber      : string      read FSerialNumber      write SetNone2;
    property FileSystemType    : string      read FFileSystemType    write SetNone2;
  end;

implementation

uses X2000About;

{$M+}
{$F+}
{$I X2000.inc}

procedure TDiskInfo2000X.SetNone(value: DWord);
begin
end;

procedure TDiskInfo2000X.SetNone2(value: String);
begin
end;

procedure TDiskInfo2000X.SetNone3(value: String30);
begin
end;

constructor TDiskInfo2000X.create(AOwner:TComponent);
begin
  inherited create(AOwner);
  setdrive('C');
end;

function TDiskInfo2000X.IsDriveReady(DriveLetter:PChar):bool;
var
  OldErrorMode : Word;
  OldDirectory : String;
begin
  OldErrorMode:=SetErrorMode(SEM_NOOPENFILEERRORBOX);
  GetDir(0, OldDirectory);
  {$I-}
   ChDir(DriveLetter+':\');
  {$I+}
  if IoResult<> 0 then
    Result:=False
  else
    Result:=True;
  ChDir(OldDirectory);
  SetErrorMode(OldErrorMode);
end;

procedure TDiskInfo2000X.SetDrive(value:string1);
    Function DecToHex( aValue : LongInt ) : String;
    Var
        w : Array[ 1..2 ] Of Word Absolute aValue;
        Function HexByte( b : Byte ) : String;
        Const
            Hex : Array[ $0..$F ] Of Char = '0123456789ABCDEF';
        Begin
            HexByte := Hex[ b Shr 4 ] + Hex[ b And $F ];
        End;
        Function HexWord( w : Word ) : String;
        Begin
            HexWord := HexByte( Hi( w ) ) + HexByte( Lo( w ) );
        End;
    Begin
        Result := HexWord( w[ 2 ] ) + HexWord( w[ 1 ] );
    End;

var fulldrive  :string[3];
    tmp_drive         :array[0..2] of char;
    Tmp_SectorsPerCluster,
    Tmp_BytesPerSector,
    Tmp_FreeClusters,
    Tmp_Clusters      :DWORD;
    VolName           :array[0..255] of Char;
    SerialN           :DWORD;
    MaxCLength        :DWORD;
    FileSysFlag       :DWORD;
    FileSysName       :array[0..255] of Char;
    Tmp_1:LONGLONG;
    Tmp_2:LONGLONG;
    Tmp_3:LONGLONG;
begin

  if not IsDriveReady(pchar(string(value))) then exit;
  fdrive:=value;
  fulldrive:=value + ':\';
  strpcopy(tmp_drive,fulldrive);

  DriveType:=GetDriveTyp;

  GetVolumeInformation(tmp_drive, VolName, 255, @SerialN, MaxCLength,
     FileSysFlag, FileSysName, 255);

  FVolumeName:=VolName;
  FSerialNumber:=DecToHex(SerialN);
  FFileSystemType:=FileSysName;

  if GetDiskFreeSpace(tmp_drive,Tmp_SectorsPerCluster,Tmp_BytesPerSector,
                       Tmp_FreeClusters,Tmp_Clusters) then
  begin
     FSectorsPerCluster:=Tmp_SectorsPerCluster;
     FBytesPerSector:=Tmp_BytesPerSector;
     FFreeClusters:=Tmp_FreeClusters;
     FClusters:=Tmp_Clusters;
  end;

  if uppercase(FileSysName)='FAT32' then
  begin
      GetDiskFreeSpaceEx(Tmp_drive, TMP_1, TMP_2, @TMP_3);
      if TMP_1>1048576 then FDiskFreeSpace  :=inttostr(TMP_1 div 1048576)+' MB';
      if TMP_2>1048576 then FTotalDiskSpace :=inttostr(TMP_2 div 1048576)+' MB';
      if TMP_1<1048576 then FDiskFreeSpace  :=inttostr(TMP_1)+' MB';
      if TMP_2<1048576 then FTotalDiskSpace :=inttostr(TMP_2)+' MB';
  end;
  if uppercase(FileSysName)<>'FAT32' then
  begin
     FDiskFreeSpace  :=inttostr((Tmp_FreeClusters*Tmp_BytesPerSector*Tmp_SectorsPerCluster) div 1048576)+' MB';
     FTotalDiskSpace :=inttostr((Tmp_Clusters*Tmp_BytesPerSector*Tmp_SectorsPerCluster) div 1048576)+' MB';
  end;
end;

function TDiskInfo2000X.GetDriveTyp :string30;
var fulldrive  :string[3];
    tmp_drive         :array[0..2] of char;
begin
  fulldrive:=fdrive + ':\';
  strpcopy(tmp_drive,fulldrive);
  case GetDriveType(tmp_drive) of
    DRIVE_UNKNOWN     :result:='No Type Information';
    DRIVE_NO_ROOT_DIR :result:='Root Directory does not exist';
    DRIVE_REMOVABLE   :result:='Removable';
    DRIVE_FIXED       :result:='Fixed';
    DRIVE_REMOTE      :result:='Remote';
    DRIVE_CDROM       :result:='CDROM';
    DRIVE_RAMDISK     :result:='RamDisk';
  end;
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
