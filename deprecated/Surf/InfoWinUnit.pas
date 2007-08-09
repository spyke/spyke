{ (c) 2000-2003 Tim Blanche, University of British Columbia }
unit InfoWinUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, Gauges, ComCtrls;

type
  TInfoWin = class(TForm)
    GroupBox1: TGroupBox;
    Label16: TLabel;
    Label8: TLabel;
    Label10: TLabel;
    lbStimTime: TLabel;
    GroupBox2: TGroupBox;
    GroupBox3: TGroupBox;
    Label9: TLabel;
    ProgressBar1: TProgressBar;
    ProgressBar2: TProgressBar;
    ProgressBar3: TProgressBar;
    ProgressBar4: TProgressBar;
    ProgressBar5: TProgressBar;
    ProgressBar6: TProgressBar;
    lbRecorded: TLabel;
    Label4: TLabel;
    Label2: TLabel;
    Label13: TLabel;
    lbAcquired: TLabel;
    Label1: TLabel;
    Label6: TLabel;
    Label3: TLabel;
    Label5: TLabel;
    Label11: TLabel;
    Label12: TLabel;
    GaugeDS: TGauge;
    Label7: TLabel;
    Label14: TLabel;
    procedure BuffLabelClick(Sender: TObject);
    procedure FileLabelClick(Sender: TObject);
    procedure lbStimTimeClick(Sender: TObject);
  private
    { Private declarations }
  public
    procedure DisableStimulusInfo;
    procedure EnableStimulusInfo;
    { Public declarations }
  end;

implementation

{$R *.DFM}


{-----------------------------------------------------------------------}
procedure TInfoWin.DisableStimulusInfo;
begin
  lbStimTime.Enabled:= False;
  Label8.Enabled:= False;
  Label10.Enabled:= False;
  Label14.Enabled:= False;
  Label16.Enabled:= False;
  GaugeDS.BackColor:= clLtGray;
end;

{-----------------------------------------------------------------------}
procedure TInfoWin.EnableStimulusInfo;
begin
  lbStimTime.Enabled:= True;
  Label8.Enabled:= True;
  Label10.Enabled:= True;
  Label14.Enabled:= True;
  Label16.Enabled:= True;
  GaugeDS.BackColor:= clWhite;
end;

{-----------------------------------------------------------------------}
procedure TInfoWin.BuffLabelClick(Sender: TObject);
begin
  lbAcquired.Tag:= (lbAcquired.Tag + 1) mod 3; //toggle 3 display states
  case lbAcquired.Tag of
    0 : begin
          lbAcquired.Hint:= 'Samples acquired';
          Label2.Hint:= lbAcquired.Hint;
        end;
    1 : begin
          lbAcquired.Hint:= 'Buffers acquired';
          Label2.Hint:= lbAcquired.Hint;
        end else
   {2}  begin
          lbAcquired.Hint:= 'MSamples acquired';
          Label2.Hint:= lbAcquired.Hint;
        end;
  end{case};
end;

{-----------------------------------------------------------------------}
procedure TInfoWin.FileLabelClick(Sender: TObject);
begin
  lbRecorded.Tag:= (lbRecorded.Tag + 1) mod 4; //cycle thru' display modes
  case lbRecorded.Tag of
    0 : lbRecorded.Hint:= 'Bytes recorded';
    1 : lbRecorded.Hint:= 'MBytes recorded';
    2 : lbRecorded.Hint:= 'GBytes recorded'
  else lbRecorded.Hint:= 'Cumulative time recorded';
  end{case};
end;

{-----------------------------------------------------------------------}
procedure TInfoWin.lbStimTimeClick(Sender: TObject);
begin
  lbStimTime.Tag:= not(lbStimTime.Tag); //toggle time display mode
  case lbStimTime.Tag of
    0 : lbStimTime.Caption:= 'Time remaining:';
   -1 : lbStimTime.Caption:= 'Time elapsed:';
  end{case};
end;

end.
