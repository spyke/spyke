program Anal3DPredict;

uses
  Forms,
  SurfLocateAndSort in 'SurfLocateAndSort.pas' {LocSortForm},
  SurfSortMain in 'SurfSortMain.pas' {SurfSortForm},
  SurfPublicTypes in '..\Public\SurfPublicTypes.pas',
  WaveFormUnit in '..\Public\WaveFormUnit.pas' {WaveFormWin},
  NumRecipies in '..\Surf\NumRecipies.pas',
  ElectrodeTypes in '..\Surf\ElectrodeTypes.pas',
  WaveFormPlotUnit in '..\Public\WaveFormPlotUnit.pas' {colo};

{$R *.RES}

begin
  Application.Initialize;
  Application.CreateForm(TSurfSortForm, SurfSortForm);
  Application.CreateForm(TLocSortForm, LocSortForm);
  Application.Run;
end.
