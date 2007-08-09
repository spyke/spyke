unit PahUnit;
interface
uses
  SysUtils, Windows, Classes, Graphics, Forms,
  Controls, Buttons, StdCtrls, ExtCtrls, Dialogs, Messages;

const
  MAXCLUSTERS = 11;
  COLOR_TABLE : array[0..MAXCLUSTERS] of TColor = (clLtGray,clFuchsia,clLime,clBlue,
                                                   clYellow,clAqua,clTeal,
                                                   clTeal,clPurple,clRed,
                                                   clGreen,clWhite);

  PORTADDRESS : array[0..1] of Word = ($378,$278);

procedure Delay (Seconds, MilliSec: Word);
procedure Gray2YB(Gray : single; var R,G,B : integer; checkbounds : boolean);
procedure ShowIMessage(i : integer);
procedure OutPort(port : word; bval : byte);

implementation

{==============================================================================}
procedure Delay (Seconds, MilliSec: Word);
var
  TimeOut: TDateTime;
begin
  TimeOut := Now + EncodeTime (0,Seconds div 60,Seconds mod 60, MilliSec);
  // wait until he TimeOut time
  while Now < TimeOut do
    Application.ProcessMessages;
end;

{==============================================================================}
procedure Gray2YB(Gray : single; var R,G,B : integer; checkbounds : boolean);
var igr : integer;
procedure check (var x : integer);
begin
  if x < 0 then x := 0;
  if x > 255 then x := 255;
end;
begin
  igr := round(Gray-128);
  R := 128 + igr; {yellow}
  B := 128 - igr; {blue}
  if checkbounds then
  begin
    check (R);
    check (B);
  end;
  G := R;                 {yellow}
end;

Procedure ShowIMessage(i : integer);
begin
  Showmessage(inttostr(i));
end;
{==============================================================================}
procedure OutPort(port : word; bval : byte);
begin
asm
  push ax  // back up ax
  push dx  // back up dx

  {direct port writes}
  mov dx,port
  mov al,bval
  out dx,al

  pop dx  // restore dx
  pop ax  // restore ax
end;
end;

end.
