program MSeqAnal;

uses
  Forms,
  MSeqAnalForm in 'MSeqAnalForm.pas' {MSeqForm};

{$R *.RES}

begin
  Application.Initialize;
  Application.CreateForm(TMSeqForm, MSeqForm);
  Application.Run;
end.
