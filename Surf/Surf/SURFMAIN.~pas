unit SURFMAIN;

interface

{$DEFINE ATLAB}

uses Windows, Classes, Graphics, Forms, Controls, Menus,
  Dialogs, StdCtrls, Buttons, ExtCtrls, ComCtrls, ImgList,
  ToolWin, Sysutils, Exec,

{$IFDEF ATLAB}
  {SURFMainAcq,} SURFContAcq,
{$ENDIF}
  About;{SurfComm;}


type
  TSurfForm = class(TForm)
    OpenDialog: TOpenDialog;
    SaveDialog: TSaveDialog;
    StatusBar: TStatusBar;
    MainMenu: TMainMenu;
    Help1: TMenuItem;
    HelpAboutItem: TMenuItem;
    Analyze: TMenuItem;
    Acquire: TMenuItem;
    Default: TMenuItem;
    Polytrode: TMenuItem;
    Continuous: TMenuItem;
    procedure AcquireClick(Sender: TObject);
    procedure PolytrodeButClick(Sender: TObject);
    procedure AnalButClick(Sender: TObject);
    procedure HelpAbout1Execute(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
  private
    { Private declarations }
    //SurfComm : TSurfComm;
    mi : array of TMenuItem;
   public
    { Public declarations }
  end;

var
  SurfForm: TSurfForm;

implementation

{$R *.DFM}

{--------------------------------------------------------------------------------}
procedure TSurfForm.FormCreate(Sender: TObject);
var path : string;
  Procedure BuildMenu(menu : TMenuItem; path : string; Action : TNotifyEvent);
  var name : string;
      SearchRec : TSearchRec;
      i : integer;
  begin
    if FindFirst(path+'*.EXE',faAnyFile, SearchRec) = 0 then
    while true do
    begin
      SetLength(mi,length(mi)+1);
      i := length(mi)-1;
      mi[i] := TMenuItem.Create(Analyze);
      name := SearchRec.Name;
      SetLength(name,length(name)-4); //remove .exe
      mi[i].Name := name;
      mi[i].Caption := name;
      mi[i].Hint := path + SearchRec.Name;
      mi[i].OnClick := Action;
      Menu.Add(mi[i]);
      if (FindNext(SearchRec) <> 0) then break;
    end;
    FindClose(SearchRec);
  end;
begin
  //Get the directory that surf resides in.
  //Look in the plugin Analysis, Acquisition, and Polytrode folders
  //Add any .EXEs to the menus that are found there
  path := ExtractFilePath(paramstr(0));
  BuildMenu(Analyze,path + 'PLUGINS\ANALYSIS\',AnalButClick); //add analysis plugins
  BuildMenu(Acquire,path + 'PLUGINS\ACQUISITION\',AcquireClick); //add acquisition plugins
  BuildMenu(Polytrode,path + 'PLUGINS\POLYTRODE\',PolytrodeButClick); //add polytrode plugins
end;

{--------------------------------------------------------------------------------}
procedure TSurfForm.HelpAbout1Execute(Sender: TObject);
begin
  AboutBox.ShowModal;
end;

{--------------------------------------------------------------------------------}
procedure TSurfForm.PolytrodeButClick(Sender: TObject);
begin
  NewExec((Sender as TMenuItem).Hint + ' SURFv1.0 ' + inttostr(Handle));
{  SurfComm.Free;
  SurfComm := TSurfComm.CreateParented(Handle);
  SurfComm.CallUserApp((Sender as TMenuItem).Hint);}
end;

{--------------------------------------------------------------------------------}
procedure TSurfForm.AnalButClick(Sender: TObject);
begin
  NewExec((Sender as TMenuItem).Hint + ' SURFv1.0 ' + inttostr(Handle));
{  SurfComm.Free;
  SurfComm := TSurfComm.CreateParented(Handle);
  SurfComm.CallUserApp((Sender as TMenuItem).Hint);}
end;

{--------------------------------------------------------------------------------}
procedure TSurfForm.AcquireClick(Sender: TObject);
begin
  Hide;
{$IFDEF ATLAB}
  if (Sender as TMenuItem).Hint = 'Continuous stream to disk' then
  begin
    Application.CreateForm(TContAcqForm, ContAcqForm);
    ContAcqForm.ShowModal;
  end else if (Sender as TMenuItem).Hint = 'Threshold-triggered acquisition' then
  begin
    {SurfAcqForm.NumUserApps := 0;
     SurfAcqForm.ShowModal;}
  end else
  begin
    {SurfAcqForm.NumUserApps := 1;
    SetLength(SurfAcqForm.UserFileNames,SurfAcqForm.NumUserApps);
    SurfAcqForm.UserFileNames[0] := (Sender as TMenuItem).Hint;}
  end;
{$ENDIF}
  Show;
end;

{--------------------------------------------------------------------------------}
procedure TSurfForm.FormClose(Sender: TObject; var Action: TCloseAction);
var i : integer;
begin
  //Free up the menu items
  for i := 0 to length(mi)-1 do mi[i].free;
end;

end.
