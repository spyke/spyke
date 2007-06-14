program TerminalPortExample;

uses
  Forms,
  TerminalPortMain in 'TerminalPortMain.pas' {MainForm};

{$R *.RES}

begin
  Application.Initialize;
  Application.Title := 'Example';
  Application.CreateForm(TMainForm, MainForm);
  Application.Run;
end.
