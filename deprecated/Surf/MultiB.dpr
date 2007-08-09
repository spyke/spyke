program MultiB;

uses
  Forms,
  Multiboard in 'Multiboard.pas' {DTConfig},
  xcorr in 'xcorr.pas' {Form5};

{$R *.RES}

begin
  Application.Initialize;
  Application.CreateForm(TForm5, Form5);
  Application.CreateForm(TDTConfig, DTConfig);
  Application.Run;
end.
