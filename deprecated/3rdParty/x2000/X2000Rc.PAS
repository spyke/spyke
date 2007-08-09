{ ****************************************************************
  Info               :  TRemoveCaptionX 
                        Freeware

  Source File Name   :  X2000Rc.PAS
  Author             :  Baldemaier Florian (Baldemaier.Florian@gmx.net)
  Compiler           :  Delphi 4.0 Client/Server, Service Pack 3
  Decription         :  Removes the caption of a Form.

  Testet on          :  Intel Pentium II, 300 Mhz and 450 Mhz, Windows 98
                        Intel Pentium III, 500 Mhz, Windows 98
                        Intel Pentium 200 Mhz, Windows98
                        D4 Service Pack 3
                        Mircosoft Windows 98
                        Mircosoft Windows 98 SE
**************************************************************** }
unit X2000RC;

interface

uses
   Windows, SysUtils, Messages, Classes, Graphics, Controls, Forms,
   Dialogs, DsgnIntf, ExtCtrls, StdCtrls, ShellAPI;

type
   TAbout2000X=class(TPropertyEditor)
   public
    procedure Edit; override;
    function GetAttributes: TPropertyAttributes; override;
    function GetValue: string; override;
   end;

   TRemoveCaption2000X = class(TComponent)
   private
     FAbout   : TAbout2000X;
     FDiff    : Integer;
     NActive  : boolean;
     FEnabled : boolean;
     procedure SetActive(Value: Boolean);
   published
     constructor create(AOwner:TComponent); override;
     destructor destroy; override;
     procedure Add(TheForm: TForm);
     procedure Remove(TheForm: TForm);
     property Enabled   : boolean      read FEnabled  write SetActive default false;
     property About     : TAbout2000X  read FAbout    write FAbout;
   end;

implementation

uses X2000about;

{$I X2000.inc}

var ParentForm:TForm;

procedure TRemoveCaption2000X.SetActive(Value: Boolean);
begin
  if value then begin
     remove(parentform);
     FEnabled:=true;
  end;
  if not value then begin
     add(parentform);
     FEnabled:=false;
  end;
end;

procedure TRemoveCaption2000X.Remove(TheForm:TForm);
begin
  NActive:=True;
  FDiff := TheForm.Height - TheForm.ClientHeight;
  SetWindowLong(TheForm.Handle, GWL_STYLE, GetWindowLong(TheForm.Handle, GWL_Style) and not WS_Caption);
  TheForm.Height:=TheForm.ClientHeight;
end;

procedure TRemoveCaption2000X.Add(TheForm:TForm);
begin
  NActive:=False;
  If Self.ComponentState = [csDestroying] Then exit;
  SetWindowLong(TheForm.Handle, GWL_STYLE, GetWindowLong(TheForm.Handle, GWL_Style) + WS_Caption);
  TheForm.Height:=TheForm.Height + FDiff;
end;

constructor TRemoveCaption2000X.create(AOwner:TComponent);
begin
  inherited create(AOwner);
  ParentForm := TForm(AOwner);
end;

destructor TRemoveCaption2000X.destroy;
begin
  if NActive then Add(ParentForm);
  inherited Destroy;
end;

procedure TAbout2000X.Edit;
begin
 with TAboutForm.Create(Application) do begin
  try
    ShowModal;
  finally
    Free;
  end;
 end;
end;

function TAbout2000X.GetAttributes: TPropertyAttributes;
begin
    Result := [paMultiSelect, paDialog, paReadOnly];
end;

function TAbout2000X.GetValue: string;
begin
    Result := '(X2000)';
end;

end.
