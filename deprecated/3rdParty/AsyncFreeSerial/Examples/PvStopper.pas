{***********************************************************}
{                                                           }
{     PvComponents v1.3                                     }
{                                                           }
{     (c) 1999 Petr Vones, email: petr.v@mujmail.cz         }
{                                                           }
{***********************************************************}

unit PvStopper;

interface

{$I PVDEFINE.INC}

uses
  Windows, Messages;

{$IFDEF PV_D4UP}

function SP_Start(TimerNum: Integer): Boolean;

function SP_Stop(TimerNum: Integer): Boolean;

function SP_Active(TimerNum: Integer): Boolean;

function SP_Time(TimerNum: Integer): Extended;

function SP_Exists: Boolean;

procedure SP_ClearAll;

{$ELSE}

// Delphi 2 and 3 doesn't support Int64 type. You can use DLL
// (PvStopperDLL.dpr) compiled under Delphi 4 or greater.

function SP_Start(TimerNum: Integer): Boolean; stdcall;

function SP_Stop(TimerNum: Integer): Boolean; stdcall;

function SP_Active(TimerNum: Integer): Boolean; stdcall;

function SP_Time(TimerNum: Integer): Extended; stdcall;

function SP_Exists: Boolean; stdcall;

procedure SP_ClearAll; stdcall;

{$ENDIF}

implementation

{$IFDEF PV_D4UP}

const
  MaxTimers = 16;

type
  TTimerRec = record
    ST, ET: Int64;
    Active: Boolean;
  end;

var
  CounterExists: Boolean;
  Freq: Int64;
  Timers: array[1..MaxTimers] of TTimerRec;

function SP_Start(TimerNum: Integer): Boolean;
begin
  if TimerNum in [1..MaxTimers] then
  begin
    Result := QueryPerformanceCounter(Timers[TimerNum].ST);
    Timers[TimerNum].Active := Result;
  end else Result := False;
end;

function SP_Stop(TimerNum: Integer): Boolean;
begin
  if TimerNum in [1..MaxTimers] then
  begin
    Result := QueryPerformanceCounter(Timers[TimerNum].ET);
    Timers[TimerNum].Active := False;
  end else Result := False;
end;

function SP_Active(TimerNum: Integer): Boolean;
begin
  if TimerNum in [1..MaxTimers] then
    Result := Timers[TimerNum].Active
  else
    Result := False;
end;

function SP_Time(TimerNum: Integer): Extended;
begin
  if TimerNum in [1..MaxTimers] then
    with Timers[TimerNum] do Result := (ET - ST) / Freq
  else
    Result := 0;
end;

function SP_Exists: Boolean;
begin
  Result := CounterExists;
end;

procedure SP_ClearAll;
begin
  FillChar(Timers, Sizeof(Timers), 0);
end;

procedure Init;
begin
  CounterExists := QueryPerformanceFrequency(Freq);
  SP_ClearAll;
end;

initialization
  Init;

{$ELSE}

const
  stopperdll = 'PvStopperDLL.dll';

function SP_Start(TimerNum: Integer): Boolean; stdcall; external stopperdll;

function SP_Stop(TimerNum: Integer): Boolean; stdcall; external stopperdll;

function SP_Active(TimerNum: Integer): Boolean; stdcall; external stopperdll;

function SP_Time(TimerNum: Integer): Extended; stdcall; external stopperdll;

function SP_Exists: Boolean; stdcall; external stopperdll;

procedure SP_ClearAll; stdcall; external stopperdll;

{$ENDIF}

end.
