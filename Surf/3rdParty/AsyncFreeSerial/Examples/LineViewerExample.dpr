program LineViewerExample;

uses
  Forms,
  LineViewerMain in 'LineViewerMain.pas' {Form1};

{$R *.RES}

begin
  Application.Initialize;
  Application.Title := 'Example';
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
