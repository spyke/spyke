program SurfBawd;

uses
  Forms,
  SurfBawdMain in 'SurfBawdMain.pas' {SurfBawdForm},
  SurfPublicTypes in '..\Public\SurfPublicTypes.pas',
  WaveFormPlotUnit in '..\Public\WaveFormPlotUnit.pas' {colo},
  SurfMathLibrary in '..\Public\SurfMathLibrary.pas',
  InfoWinUnit in '..\Public\InfoWinUnit.pas' {InfoWin},
  SurfLocateAndSort in '..\Anal3DPredict\SurfLocateAndSort.pas' {LocSortForm},
  RasterPlotUnit in '..\Public\RasterPlotUnit.pas' {RasterForm},
  ElectrodeTypes in '..\Surf\ElectrodeTypes.pas',
  FileProgressUnit in '..\Surf\FileProgressUnit.pas' {FileProgressWin},
  PolytrodeGUI in 'PolytrodeGUI.pas' {PolytrodeGUIForm},
  TemplateFormUnit in '..\Surf\TemplateFormUnit.pas' {TemplateWin},
  ChartFormUnit in '..\Public\ChartFormUnit.pas' {ChartWin},
  HistogramUnit in '..\Surf\HistogramUnit.pas' {HistogramWin};

{$R *.RES}

begin
  Application.Initialize;
  Application.Title := 'SurfBawd';
  Application.CreateForm(TSurfBawdForm, SurfBawdForm);
  Application.Run;
end.
