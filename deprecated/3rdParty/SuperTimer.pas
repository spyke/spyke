{***********************************************************}
{                                                           }
{        TSuperTimer Component                              }
{                                                           }
{        Copyright (C) 1996, by Jan Goyvaerts               }
{        All rights reserved                                }
{                                                           }
{***********************************************************}

{***********************************************************}
{                                                           }
{        Delphi 32-bit component unit                       }
{           - TSuperTimer                                   }
{        -> JG page                                         }
{                                                           }
{***********************************************************}

{***********************************************************}
{                                                           }
{         This component has been registered to             }
{         P&M Research Technologies, Inc.                   }
{                                                           }
{***********************************************************}

{***********************************************************}
{                                                           }
{ The person or company this component is registered to,    }
{ may create applications using this component and          }
{ distribute and sell those without extra charge.           }
{ This source code must not be distributed, not in whole    }
{ nor in part, whether it has been modified or not.         }
{ This source code must not be used for developing any      }
{ product that might be considered as competitive to the    }
{ product this source originally belonged to.               }
{                                                           }
{***********************************************************}

unit SuperTimer;

{$B-,Q-,R-,S-}
{$D-,L-,Y-}

interface

uses
  Windows, Messages, MMSystem, SysUtils, Classes, Graphics, Controls, Forms, Dialogs;

type
  TSuperTimer = class;

  TTimerThread = class(TThread)                  // *** Thread used by TSuperTimer
  protected
    OwnerTimer: TSuperTimer;
    procedure Execute; override;
  end;

  TSuperTimer = class(TComponent)                // *** Highly accurate and extended version of the TTimer component
  private
    FEnabled: Boolean;                                     // Property values
    FInterval: Word;
    FOnTimer: TNotifyEvent;
    FThreadPriority: TThreadPriority;
    FNowMS: Longint;
    FCountDown: Boolean;
    FStopTimer: Longint;
    FOnStop: TNotifyEvent;
    FTimerThread: TTimerThread;                            // Our timer thread
    ThreadNextTime: Longint;                               // Used inside TTimerThread.Execute
    procedure SetEnabled(Value: Boolean);                  // Property read/write routines
    procedure SetInterval(Value: Word);
    procedure SetThreadPriority(Value: TThreadPriority);
    procedure SetNowMS(Value: Longint);
    function GetNowTimer: Longint;
    procedure SetNowTimer(Value: Longint);
    function GetNow: TDateTime;
    procedure SetNow(Value: TDateTime);
    function GetStopMS: Longint;
    procedure SetStopMS(Value: Longint);
    function GetStop: TDateTime;
    procedure SetStop(Value: TDateTime);
    procedure SetCountDown(Value: Boolean);
  protected
    StartNow: Longint;                                     // Value of NowMS when Enabled became True
    StartTime: Longint;                                    // System time (timeGetTime) when Enabled became True
    procedure Timer; dynamic;                              // Like TTimer.Timer: calls OnTimer
  public                                                      // also updates NowMS and checks for Stop
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    // Current timer time in milliseconds. Automatically increased by timer thread
    property NowMS: Longint read FNowMS write SetNowMS;
    // Amount of times the OnTimer event has (should have) been triggered until now
    property NowTimer: Longint read GetNowTimer write SetNowTimer;
    // Current timer time in TDateTime format
    property Now: TDateTime read GetNow write SetNow;
    // StopTimer in milliseconds
    property StopMS: Longint read GetStopMS write SetStopMS;
    // Timer becomes disabled automatically if NowTimer >= Stop. No stopping if Stop < 0 (default)
    property StopTimer: Longint read FStopTimer write FStopTimer;
    // StopTimer in TDateTime format
    property Stop: TDateTime read GetStop write SetStop;
  published
    // TTimer-like properties
    property Enabled: Boolean read FEnabled write SetEnabled default False;
    property Interval: Word read FInterval write SetInterval default 1000;
    property OnTimer: TNotifyEvent read FOnTimer write FOnTimer;
    // Priority of the timer's thread. Adjust as needed.
    property ThreadPriority: TThreadPriority read FThreadPriority write SetThreadPriority default tpHigher;
    // When True, NowMS decreases instead of increasing. If NowTimer <= Stop or NowTimer <= 0 the timer stops.
    property CountDown: Boolean read FCountDown write SetCountDown default False;
    // Event triggered when timer stops automatically
    property OnStop: TNotifyEvent read FOnStop write FOnStop;
  end;

procedure Register;

implementation


         { ********* TTimerThread thread ********* }

procedure TTimerThread.Execute;
var
  NowTime: Longint;
begin
  Priority := OwnerTimer.FThreadPriority;
  with OwnerTimer do
    repeat
      if FInterval > 50 then SleepEx(FInterval-50, False);
      repeat
        NowTime := timeGetTime
      until NowTime >= ThreadNextTime;
      ThreadNextTime := NowTime + FInterval;
      if FCountDown then FNowMS := StartNow - (NowTime - StartTime)
        else FNowMS := NowTime - StartTime + StartNow;
      Synchronize(Timer)
    until Terminated
end;


         { ********* TSuperTimer component ********* }

      { ****** TSuperTimer private methods ****** }

procedure TSuperTimer.SetEnabled(Value: Boolean);
begin
  if Value <> FEnabled then begin
    FEnabled := Value;
    if Value then begin
      StartNow := FNowMS;
      StartTime := timeGetTime;
      ThreadNextTime := StartTime + FInterval;
      FTimerThread.Resume
    end
    else
      FTimerThread.Suspend
  end
end;

procedure TSuperTimer.SetInterval(Value: Word);
begin
  if Value <> FInterval then begin
    if Value = 0 then Enabled := False;
    FInterval := Value
  end
end;

procedure TSuperTimer.SetThreadPriority(Value: TThreadPriority);
begin
  if Value <> FThreadPriority then begin
    FThreadPriority := Value;
    FTimerThread.Priority := Value
  end
end;

procedure TSuperTimer.SetNowMS(Value: Longint);
begin
  if Enabled then StartNow := (Value - FNowMS) + StartNow
   else FNowMS := Value
end;

function TSuperTimer.GetNowTimer: Longint;
begin
  Result := NowMS div FInterval
end;

procedure TSuperTimer.SetNowTimer(Value: Longint);
begin
  SetNowMS(Value * FInterval)
end;

function TSuperTimer.GetNow: TDateTime;
begin
  Result := NowMS / 86400000.0;
end;

procedure TSuperTimer.SetNow(Value: TDateTime);
begin
  SetNowMS(Trunc(Value * 86400000.0))
end;

function TSuperTimer.GetStopMS: Longint;
begin
  Result := FStopTimer * FInterval
end;

procedure TSuperTimer.SetStopMS(Value: Longint);
begin
  FStopTimer := Value div FInterval
end;

function TSuperTimer.GetStop: TDateTime;
begin
  Result := (FStopTimer * FInterval) / 86400000.0
end;

procedure TSuperTimer.SetStop(Value: TDateTime);
begin
  FStopTimer := Trunc(Value * 86400000.0) div FInterval
end;

procedure TSuperTimer.SetCountDown(Value: Boolean);
begin
  if FCountDown <> Value then begin
    FCountDown := Value;
    if Enabled then begin
      StartTime := timeGetTime;
      StartNow := FNowMS;
    end
  end
end;


      { ****** TSuperTimer protected methods ****** }

procedure TSuperTimer.Timer;
begin
  if Assigned(FOnTimer) then FOnTimer(Self);
  if FCountDown then
    if (NowTimer <= FStopTimer) or (FNowMS <= 0) then begin
      Enabled := False;
      if Assigned(FOnStop) then FOnStop(Self)
    end else
  else
    if (NowTimer >= FStopTimer) and (FStopTimer > 0) then begin
      Enabled := False;
      if Assigned(FOnStop) then FOnStop(Self)
    end
end;


      { ****** TSuperTimer public methods ****** }

constructor TSuperTimer.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FInterval := 1000; FStopTimer := -1;
  FThreadPriority := tpHigher;
  FTimerThread := TTimerThread.Create(True);
  FTimerThread.OwnerTimer := Self
end;

destructor TSuperTimer.Destroy;
begin
  Enabled := False;
  FTimerThread.Free;
  inherited Destroy;
end;


         { ********* Unit support routines ********* }

procedure Register;
begin
  RegisterComponents('JG', [TSuperTimer]);
end;

end.
