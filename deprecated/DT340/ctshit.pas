unit ctshit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, OleCtrls, DTAcq32Lib_TLB, DTxPascal;

type
  TForm4 = class(TForm)
    Button1: TButton;
    ListBox1: TListBox;
    Button2: TButton;
    procedure Button1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure Button2Click(Sender: TObject);
  private
    TimeBefore, TimeNow : integer;
    DT32BitCounter, DTClock: TDTAcq32;{ Private declarations }
  public
    { Public declarations }
  end;

var
  Form4: TForm4;

implementation

{$R *.DFM}

procedure TForm4.Button1Click(Sender: TObject);
begin
  TimeNow:= DT32BitCounter.CTReadEvents;
  ListBox1.Items.Add('Interval:'+inttostr(TimeNow-TimeBefore)+'ms');
  TimeBefore:= TimeNow;
end;

procedure TForm4.FormCreate(Sender: TObject);
begin
        DT32BitCounter:= TDTAcq32.Create(Self); //32 bit counter for precision clock
        with DT32BitCounter do
        begin
          Board:= 'DT340';
          SubSysType:= OLSS_CT;
          SubSysElement:= 1;
          CascadeMode:= OL_CT_CASCADE; //internally cascaded clocks
          ClockSource:= OL_CLK_EXTERNAL;//output from CT0 --> input of CT1
          CTMode:= OL_CTMODE_COUNT;
          GateType:= OL_GATE_NONE; //software start/stop
        end;
        DTClock:= TDTAcq32.Create(Self); //precision clock
        with DTClock do
        begin
          Board:= 'DT340';
          SubSysType:= OLSS_CT;
          SubSysElement:= 0;
          ClockSource:= OL_CLK_INTERNAL;
          Frequency:= 1000;
          GateType:= OL_GATE_NONE; //software start/stop
          CTMode:= OL_CTMODE_RATE;
        end;
        try
          DT32BitCounter.Config;
          DTClock.Config;
          DT32BitCounter.Start;
          DTClock.Start;
        except
          DT32BitCounter.Free;
          DTClock.Free;
        end;
end;

procedure TForm4.Button2Click(Sender: TObject);
begin
  ListBox1.Visible:= not(ListBox1.Visible);
end;

end.
