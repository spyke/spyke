program Lynx8Control;

uses
  Forms,
  EnterGain in 'EnterGain.pas' {EnterGainForm},
  L8ABOUT in 'L8about.pas' {L8AboutBox},
  L8Setup in 'L8Setup.pas' {SetupL8},
  Lynx8 in 'Lynx8.pas' {Lynx8Form},
  PahUnit in '..\Surf\PahUnit.pas',
  DTxPascal in '..\Surf\DTxPascal.pas',
  Lynx8CControl in 'Lynx8CControl.pas' {Lynx8Amp};

{$R *.RES}

begin
  Application.Initialize;
  Application.CreateForm(TLynx8Form, Lynx8Form);
  Application.CreateForm(TEnterGainForm, EnterGainForm);
  Application.CreateForm(TL8AboutBox, L8AboutBox);
  Application.CreateForm(TLynx8Amp, Lynx8Amp);
  Application.Run;
end.
 
