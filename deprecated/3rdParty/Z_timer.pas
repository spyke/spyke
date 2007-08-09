// ZTimer component, by Antonie Baars
//modified by PAH to remove timeout, and change base time to 10th ms.

// email d950021@icpc00.icpc.fukui-u.ac.jp

unit Z_timer;

interface
uses classes,windows,messages,forms;

TYPE

TZTimer = class(Tcomponent)
 private
    FOnTimer: TNotifyEvent;
    FEnabled: Boolean;
    fcount,FInterval,ReqToStop:integer;
    FWindowHandle: HWND;
    procedure UpdateTimer;
    procedure SetEnabled(Value: Boolean);
    procedure SetInterval(Value: extended);
    function  getinterval:extended;
    //procedure SetInterval(Value: Int64{extended});
    //function  getinterval:Int64{extended};
    procedure SetOnTimer(Value: TNotifyEvent);
    procedure Timerloop;
    procedure WndProc(var Mesg: TMessage);
  protected
    procedure Timer;virtual;
  public
    property count :integer read fcount;
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
  published
    property Enabled: Boolean read FEnabled write SetEnabled default True;
    property Interval: extended read getInterval write SetInterval;
    //property Interval: Int64{extended} read getInterval write SetInterval;
    property OnTimer: TNotifyEvent read FOnTimer write SetOnTimer;
  end;

procedure Register;

implementation

const      WM_PACER:integer=WM_USER+202;

procedure Register;
begin
  RegisterComponents('System', [TZTimer]);
end;

constructor TZTimer.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FEnabled := false;
  FInterval := 10000;
  FWindowHandle := AllocateHWnd(WndProc);
end;

destructor TZTimer.Destroy;
begin
  FEnabled := False;
  updatetimer;
  DeallocateHWnd(FWindowHandle);
  inherited Destroy;
end;

procedure TZTimer.WndProc(var Mesg: TMessage);
begin
  with Mesg do
    if Msg = WM_PACER then
      try
        Timerloop;
      except
        Application.HandleException(Self);
      end
    else
      Result := DefWindowProc(FWindowHandle, Msg, wParam, lParam);
end;

procedure TZTimer.UpdateTimer;
begin
reqtostop:=1;                                   // break out current loop
if (FInterval <> 0) and FEnabled and Assigned(FOnTimer)
   and not (csdesigning in componentstate) then
   postmessage(fwindowhandle,WM_PACER,0,0);     // start new run after cleanup
end;


procedure TZTimer.SetEnabled(Value: Boolean);
begin
  if Value <> FEnabled then
  begin
    FEnabled := Value;
    UpdateTimer;
  end;
end;

//procedure TZTimer.SetInterval(Value: Int64{extended});
procedure TZTimer.SetInterval(Value: extended);
begin
    FInterval := round(Value);
    //FInterval := Value;
    if finterval<1 then finterval:=1;
    UpdateTimer;
end;

//function TZTimer.getInterval: Int64{extended};
function TZTimer.getInterval: extended;
begin
result:=FInterval;
end;

procedure TZTimer.SetOnTimer(Value: TNotifyEvent);
begin
  FOnTimer := Value;
  UpdateTimer;
end;

procedure TZTimer.Timer;
begin
  if Assigned(FOnTimer) then FOnTimer(Self);
end;
procedure TZTimer.Timerloop;
label again;
var sampletime,areqtostop:integer;
begin
sampletime:=round(finterval/0.1509);
reqtostop:=0;
fcount:=0;
{$IFDEF WIN32}
SetPriorityClass(GetCurrentProcess(),REALTIME_PRIORITY_CLASS);
{$ENDIF}
asm
        in   al,61h
        and  al,0010000b
        mov  ah,al
again:  mov  ecx,sampletime
@wait:  in   al,61h
        and  al,0010000b
        cmp  al,ah
        je   @wait         // wait for levelchange
        mov  ah,al
        dec  ecx
        jnz  @wait
        push ax
        end;
        inc(fcount);
        timer ;        // perform ontimer event
        //if fcount>timeout then reqtostop:=1;
        areqtostop:=reqtostop;
        asm
        pop ax
        cmp  areqtostop,0
        jz   again
        end;
{$IFDEF WIN32}
SetPriorityClass(GetCurrentProcess(),NORMAL_PRIORITY_CLASS);
{$ENDIF}
end;


end.
