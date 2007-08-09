{==============================================================================|
| Project : Delphree - AsyncFree                                 | 001.002.000 |
|==============================================================================|
| Content:  Common functions and classes used in AsyncFree                     |
|==============================================================================|
| The contents of this file are subject to the Mozilla Public License Ver. 1.0 |
| (the "License"); you may not use this file except in compliance with the     |
| License. You may obtain a copy of the License at http://www.mozilla.org/MPL/ |
|                                                                              |
| Software distributed under the License is distributed on an "AS IS" basis,   |
| WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License for |
| the specific language governing rights and limitations under the License.    |
|==============================================================================|
| The Original Code is AsyncFree Library.                                      |
|==============================================================================|
| The Initial Developer of the Original Code is Petr Vones (Czech Republic).   |
| Portions created by Petr Vones are Copyright (C) 1998, 1999.                 |
| All Rights Reserved.                                                         |
|==============================================================================|
| Contributor(s):                                                              |
|==============================================================================|
| History:                                                                     |
|   see AfRegister.pas                                                         |
|==============================================================================}

unit AfUtils;

{$I PVDEFINE.INC}

interface

uses
  Classes, Windows;

type
  PRaiseFrame = ^TRaiseFrame;
  TRaiseFrame = record
    NextRaise: PRaiseFrame;
    ExceptAddr: Pointer;
    ExceptObject: TObject;
    ExceptionRecord: PExceptionRecord;
  end;

  procedure SafeCloseHandle(var Handle: THandle);

  procedure ExchangeInteger(X1, X2: Integer);

  procedure FillInteger(const Buffer; Size, Value: Integer);

  function LongMulDiv(Mult1, Mult2, Div1: Longint): Longint; stdcall;

{$IFDEF PV_D2}
  function CompareMem(P1, P2: Pointer; Length: Integer): Boolean;
{$ENDIF}

implementation

procedure SafeCloseHandle(var Handle: THandle);
begin
  if (Handle <> INVALID_HANDLE_VALUE) and CloseHandle(Handle) then
    Handle := INVALID_HANDLE_VALUE;
end;

procedure ExchangeInteger(X1, X2: Integer); register; assembler;
asm
        XCHG EAX, EDX 
end;


procedure FillInteger(const Buffer; Size, Value: Integer); register; assembler;
asm
        PUSH EDI
        MOV  EDI, EAX
        XCHG ECX, EDX
        MOV  EAX, EDX
        REP  STOSD
        POP  EDI
end;

function LongMulDiv(Mult1, Mult2, Div1: Longint): Longint; stdcall;
  external 'kernel32.dll' name 'MulDiv';

{$IFDEF PV_D2}
function CompareMem(P1, P2: Pointer; Length: Integer): Boolean; assembler;
asm
        PUSH    ESI
        PUSH    EDI
        MOV     ESI,P1
        MOV     EDI,P2
        MOV     EDX,ECX
        XOR     EAX,EAX
        AND     EDX,3
        SHR     ECX,2
        REPE    CMPSD
        JNE     @@2
        MOV     ECX,EDX
        REPE    CMPSB
        JNE     @@2
@@1:    INC     EAX
@@2:    POP     EDI
        POP     ESI
end;
{$ENDIF}

end.
