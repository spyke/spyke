unit LineViewerMain;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, StdCtrls, ComCtrls, AfViewers;

type
  TForm1 = class(TForm)
    Panel1: TPanel;
    ChangeFontBtn: TButton;
    FontDialog: TFontDialog;
    OptionsGroup: TRadioGroup;
    ColorsGroup: TRadioGroup;
    CopyBtn: TButton;
    FontCacheCheckBox: TCheckBox;
    AfLineViewer: TAfLineViewer;
    procedure AfLineViewerGetText(Sender: TObject; Line: Integer;
      var Text: String; var ColorMode: TAfCLVColorMode;
      var CharColors: TAfCLVCharColors);
    procedure ChangeFontBtnClick(Sender: TObject);
    procedure OptionsGroupClick(Sender: TObject);
    procedure ColorsGroupClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure CopyBtnClick(Sender: TObject);
    procedure FontCacheCheckBoxClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.DFM}

procedure TForm1.AfLineViewerGetText(Sender: TObject; Line: Integer;
  var Text: String; var ColorMode: TAfCLVColorMode;
  var CharColors: TAfCLVCharColors);
var
  I: Integer;
begin
  Text := Format('This is line number: %d', [Line]);
  case ColorsGroup.ItemIndex of
    0: ColorMode := cmDefault;
    1: begin
         ColorMode := cmLine;
         CharColors[0].FColor := Line * 5000 or 5000;
         CharColors[0].BColor := clBlack;
         CharColors[0].Style := AfLineViewer.Font.Style;
       end;
    2: begin
         ColorMode := cmChars;
         for I := 0 to AfLineViewer.MaxLineLength - 1 do
         begin
           CharColors[I].FColor := Line * I * 5000 or 5000;
           CharColors[I].BColor := Line div 4 * I;

           CharColors[I].BColor := clwhite;
           if I < Length(Text) then
             CharColors[I].Style := TFontStyles(Byte(I) and $0F)
           else
             CharColors[I].Style := [];
         end;
       end;
  end;
end;

procedure TForm1.ChangeFontBtnClick(Sender: TObject);
begin
  if FontDialog.Execute then
    AfLineViewer.Font.Assign(FontDialog.Font);
  AfLineViewer.SetFocus;  
end;

procedure TForm1.OptionsGroupClick(Sender: TObject);
begin
  with AfLineViewer do
  begin
    case OptionsGroup.ItemIndex of
      0: Options := Options - [loShowLineCursor, loDrawFocusSelect] + [loShowCaretCursor];
      1: Options := Options + [loShowLineCursor, loDrawFocusSelect] - [loShowCaretCursor];
    end;
    SetFocus;
  end;
end;

procedure TForm1.ColorsGroupClick(Sender: TObject);
begin
  case ColorsGroup.ItemIndex of
    0: AfLineViewer.Color := clWindow;
    1, 2: AfLineViewer.Color := clBlack;
  end;
  AfLineViewer.Invalidate;
  AfLineViewer.SetFocus;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  AfLineViewer.Color := clBlack;
end;

procedure TForm1.CopyBtnClick(Sender: TObject);
begin
  AfLineViewer.CopyToClipboard;
end;

procedure TForm1.FontCacheCheckBoxClick(Sender: TObject);
begin
  AfLineViewer.UseFontCache := FontCacheCheckBox.Checked;
  AfLineViewer.SetFocus;
end;


end.
