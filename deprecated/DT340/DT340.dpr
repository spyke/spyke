program DT340;

uses
  Forms,
  DT340Beta in 'DT340Beta.pas' {Form1},
  DTMUX in '..\..\..\Program Files\Data Translation\Dtx-EZ\examples\vb\DIO\DTMUX.pas' {Form2},
  diskstream in 'diskstream.pas' {Form3},
  ctshit in 'ctshit.pas' {Form4};

{$R *.RES}

begin
  Application.Initialize;
  Application.CreateForm(TForm4, Form4);
  Application.Run;
end.
