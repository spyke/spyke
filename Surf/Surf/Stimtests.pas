unit Stimtests;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, OleCtrls, DTAcq32Lib_TLB, DTxPascal;

type
  TForm1 = class(TForm)
    DTDIN: TDTAcq32;
    Label1: TLabel;
    Label2: TLabel;
    procedure FormShow(Sender: TObject);
    procedure DTDINSSEventDone(Sender: TObject; var lStatus: Integer);
  private
    interrupts, now, before: integer;
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.DFM}

procedure TForm1.FormShow(Sender: TObject);
begin
  DTDIN.Subsystem:= OLSS_DIN;
  DTDIN.SubsysElement:= 3;
  DTDIN.DataFlow:= OL_DF_CONTINUOUS;
  DTDIN.Resolution:= 8;
  DTDIN.Config;
  DTDIN.Start;
  Before:= GetTickCount;
  interrupts:= 1;
end;

procedure TForm1.DTDINSSEventDone(Sender: TObject; var lStatus: Integer);
begin
  //Now:= GetTickCount;
  //Label2.Caption:= inttostr((now-before) div 10);
  //Before:= Now;
  if interrupts mod(500) = 0 then
  begin
    Now:= GetTickCount;
    Label1.Caption:= inttostr(round(interrupts /(now - before)*1000))+'Hz';
    Before:= Now;
    interrupts:= 0;
  end;
  inc(interrupts);
end;

end.
