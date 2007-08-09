program TestGIF;

uses
  Forms,
  Main in 'Main.pas' {MainForm};

begin
  Application.CreateForm(TMainForm, MainForm);
  Application.Run;
end.
