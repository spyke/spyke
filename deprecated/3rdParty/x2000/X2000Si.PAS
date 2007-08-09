{ ****************************************************************
  Info               :  TSystemInfo2000X
                        Freeware

  Source File Name   :  X2000SI.PAS
  Author             :  Baldemaier Florian (Baldemaier.Florian@gmx.net)
  Compiler           :  Delphi 4.0 Client/Server, Service Pack 3
  Operating System   :  only Windows 98 (Important!!!!)
  Decription         :  Systeminfo component. All returned information
                        are from the Windows registry.

  Testet on          :  Intel Pentium II, 300 Mhz and 450 Mhz, Windows 98
                        Intel Pentium III, 500 Mhz, Windows 98
                        Intel Pentium 200 Mhz, Windows98
                        D4 Service Pack 3
                        Mircosoft Windows 98
                        Mircosoft Windows 98 SE
**************************************************************** }

unit X2000SI;

interface

uses Windows, SysUtils, Messages, Classes, Graphics, Controls, Forms,
     Dialogs, Registry, DsgnIntf, ExtCtrls, StdCtrls, ShellAPI, Printers;

type string1 = string[1];
type string30 = string[30];
type

 TAbout2000X=class(TPropertyEditor)
 public
    procedure Edit; override;
    function GetAttributes: TPropertyAttributes; override;
    function GetValue: string; override;
 end;

 //=== THARDWAREX Classes ==================================================
 	 TMotherboardX=class (TPersistent)
	  private
	     FBiosDate, FBiosName, FBiosVer:string;
	     FBusType, FMachineType:string;
	  published
	     property BiosDate   : string read FBiosDate    write FBiosDate;
	     property BiosName   : string read FBiosName    write FBiosName;
	     property BiosVer    : string read FBiosVer     write FBiosVer;
	     property BusType    : string read FBusType     write FBusType; // AW - 12/07/1999
	     property MachineType: string read FMachineType write FMachineType; // AW - 12/07/1999
	  end;

	  TControllerX=class (TPersistent)
	  private
	    FControll:TStrings;
	    FSysC    :TStrings;
	  published
	    property Controller        : TStrings read FControll write FControll; // AW - 14/07/1999
	    property System_Components : TStrings read FSysC     write FSysC; // AW - 14/07/1999
	  end;

	  TCpuX=class (TPersistent)
	  private
	     FProcessorSpeed, FNumberOfProcessors, FActiveProcessor, FProcessor, FProcessorVendor, FProcessorMMX:string;
	     FProcessorRevision:string; // AW - 13/07/1999
	     FCoPro:string;
	  published
	     property Processor          : string read FProcessor          write FProcessor;
	     property ProcessorSpeed     : string read FProcessorSpeed     write FProcessorSpeed; // AW - 12/07/1999
	     property ProcessorVendor    : string read FProcessorVendor    write FProcessorVendor;
	     property ProcessorRevision  : string read FProcessorRevision  write FProcessorRevision; // AW - 13/07/1999
	     property ProcessorMMX       : string read FProcessorMMX       write FProcessorMMX;
	     property NumberOfProcessors : String read FNumberOfProcessors write FNumberOfProcessors; // AW - 12/07/1999
	     property ActiveProcessor    : String read FActiveProcessor    write FActiveProcessor; // AW - 12/07/1999
	     property CoProcessor        : String read FCoPro              write FCoPro;
	  end;

	  TCapabilitiesX=class (TPersistent)
	  private
             FRasterCapabilities    :TStrings;
             FCurveCapabilities     :TStrings;
             FLineCapabilities      :TStrings;
             FPolygonalCapabilities :TStrings;
             FTextCapabilities      :TStrings;
          published
             property RasterCapabilities    :TStrings read FRasterCapabilities;
             property CurveCapabilities     :TStrings read FCurveCapabilities;
             property LineCapabilities      :TStrings read FLineCapabilities;
             property PolygonalCapabilities :TStrings read FPolygonalCapabilities;
             property TextCapabilities      :TStrings read FTextCapabilities;
          end;

	  TDisplayX=class (TPersistent)
	  private
	     FDisplayDrv, FDisplayVXD, FDisplayDPI, FDisplayResolution, FDisplayBpp, FMonitor:string;
	     FModes        :TStrings;
             FCapabilities :TCapabilitiesX;
	  published
	     property Display_Driver_Description : string         read FDisplayDrv        write FDisplayDrv;
	     property Display_Driver             : string         read FDisplayVXD        write FDisplayVXD;
	     property DisplayDPI                 : string         read FDisplayDPI        write FDisplayDPI;
	     property DisplayResolution          : string         read FDisplayResolution write FDisplayResolution;
	     property DisplayBpp                 : string         read FDisplayBpp        write FDisplayBpp;
	     property Monitor                    : String         read FMonitor           write FMonitor;
	     property Modes                      : TStrings       read FModes             write FModes; // AW - 15/07/1999
	     property Capabilities               : TCapabilitiesX read FCapabilities      write FCapabilities;
	  end;

	  TKernelX=class(TPersistent)
	  private
	     FCPUUsage, FNumberOfThreads, FNumberOfDOSVirtualMachines : LongInt;
	  published
	     property CPUUsage                   : LongInt read FCPUUsage                   write FCPUUsage; // AW - 13/07/1999
	     property NumberOfThreads            : LongInt read FNumberOfThreads            write FNumberOfThreads; // AW - 13/07/1999
	     property NumberOfDOSVirtualMachines : LongInt read FNumberOfDOSVirtualMachines write FNumberOfDOSVirtualMachines; // AW - 13/07/1999
	  end;

	  TVFATX=class(TPersistent)
	  private
	     FBytesReadPerSecond, FBytesWrittenPerSecond, FDirty, FReadsPerSecond, FWritesPerSecond : LongInt;
	  published
	     property BytesReadPerSecond    : LongInt read FBytesReadPerSecond    write FBytesReadPerSecond; // AW - 13/07/1999
	     property BytesWrittenPerSecond : LongInt read FBytesWrittenPerSecond write FBytesWrittenPerSecond; // AW - 13/07/1999
	     property Dirty                 : LongInt read FDirty                 write FDirty; // AW - 13/07/1999
	     property ReadsPerSecond        : LongInt read FReadsPerSecond        write FReadsPerSecond; // AW - 13/07/1999
	     property WritesPerSecond       : LongInt read FWritesPerSecond       write FWritesPerSecond; // AW - 13/07/1999
	  end;

	  TCOMInterfaceX=class(TPersistent)
	  private
	     FAvailable : Boolean;
	     FBaudrate  : String;
	     FMaxOutputBuffer, FMaxInputBuffer, FPacketLength, FPacketVersion : LongInt;
	  published
	     property Available       : Boolean read FAvailable       write FAvailable; // AW - 13/07/1999
	     property Baudrate        : String  read FBaudrate        write FBaudrate; // AW - 13/07/1999
	     property MaxOutputBuffer : LongInt read FMaxOutputBuffer write FMaxOutputBuffer; // AW - 13/07/1999
	     property MaxInputBuffer  : LongInt read FMaxInputBuffer  write FMaxInputBuffer; // AW - 13/07/1999
	     property PacketLength    : LongInt read FPacketLength    write FPacketLength; // AW - 13/07/1999
	     property PacketVersion   : LongInt read FPacketVersion   write FPacketVersion; // AW - 13/07/1999
	  end;

	  TInterfacesX=class(TPersistent)
	  private
	     FCOM1, FCOM2, FCOM3, FCOM4 : TCOMInterfaceX;
	     FLPT1Available, FLPT2Available, FLPT3Available, FLPT4Available : Boolean;
	  published
	     property COM1 : TCOMInterfaceX read FCOM1 write FCOM1; // AW - 13/07/1999
	     property COM2 : TCOMInterfaceX read FCOM2 write FCOM2; // AW - 13/07/1999
	     property COM3 : TCOMInterfaceX read FCOM3 write FCOM3; // AW - 13/07/1999
	     property COM4 : TCOMInterfaceX read FCOM4 write FCOM4; // AW - 13/07/1999
	     property LPT1_Available : Boolean read FLPT1Available write FLPT1Available; // AW - 13/07/1999
	     property LPT2_Available : Boolean read FLPT2Available write FLPT2Available; // AW - 13/07/1999
	     property LPT3_Available : Boolean read FLPT3Available write FLPT3Available; // AW - 13/07/1999
	     property LPT4_Available : Boolean read FLPT4Available write FLPT4Available; // AW - 13/07/1999
	  end;

	  TMouseX=class(TPersistent)
	  private
	     FAvailable, FButtonsExchanged, FMouseWheelAvailable : Boolean;
	     FMouseSpeed, FDoubleClickSpeed, FDescription        : String;
	     FNumberOfButtons                                    : Integer;
	  published
	     property Available           : Boolean read FAvailable           write FAvailable; // AW - 13/07/1999
	     property Description         : String  read FDescription         write FDescription; // AW - 13/07/1999
	     property NumberOfButtons     : Integer read FNumberOfButtons     write FNumberOfButtons; // AW - 13/07/1999
	     property ButtonsExchanged    : Boolean read FButtonsExchanged    write FButtonsExchanged; // AW - 13/07/1999
	     property MouseWheelAvailable : Boolean read FMouseWheelAvailable write FMouseWheelAvailable; // AW - 13/07/1999
	     property DoubleClickSpeed    : String  read FDoubleClickSpeed    write FDoubleClickSpeed;
	     property MouseSpeed          : String  read FMouseSpeed          write FMouseSpeed;
	  end;

	  TKeyBoardX=class(TPersistent)
	  private
	     FKey, FNumberOfFunctionKeys:string;
	  published
	     property Keyboard             : String read FKey                  write FKey;
	     property NumberOfFunctionKeys : String read FNumberOfFunctionKeys write FNumberOfFunctionKeys; // AW - 12/07/1999
	  end;

	  TInputX=class(TPersistent)
	  private
             FKeyBoard: TKeyBoardX;
	     FMouse   : TMouseX;
	  published
	     property Mouse    : TMouseX    read FMouse    write FMouse;
	     property KeyBoard : TKeyBoardX read FKeyBoard write FKeyBoard;
	  end;

	  TMultiMediaX=class(TPersistent)
	  private
	     FModem, FMixer:string;
	     FMedia, FMediaControll, FAudioCompress, FImageCompress, FWave, FMidi:TStrings;
	  published
	     property Modem          : String   read FModem         write FModem;
	     property Media          : TStrings read FMedia         write FMedia;
	     property Media_Controll : TStrings read FMediaControll write FMediaControll;
	     property Audio_Compress : TStrings read FAudioCompress write FAudioCompress;
	     property Image_Compress : TStrings read FImageCompress write FImageCompress;
	     property Wave           : TStrings read FWave          write FWave;
	     property Midi           : TStrings read FMidi          write FMidi;
	     property Mixer          : String   read FMixer         write FMixer;
	  end;

	  TPrinterX=class(TPersistent)
	  private
	     FPrinter, FPort:string;
	  published
	     property Printer: String read FPrinter write FPrinter;
	     property Port   : String read FPort    write FPort;
	  end;

	  TProxyX=class(TPersistent)
	  private
	    FFTP, FGopher, FHTTP, FHTTPS, FSOCKS, FProxyOverride:string;
	  published
	    property FTP           : String read FFTP           write FFTP;
	    property GOPHER        : String read FGOPHER        write FGOPHER;
	    property HTTP          : String read FHTTP          write FHTTP;
	    property HTTPS         : String read FHTTPS         write FHTTPS;
	    property SOCKS         : String read FSOCKS         write FSOCKS;
	    property ProxyOverride : String read FProxyOverride write FProxyOverride;
	  end;

	  TNetX=class(TPersistent)
	  private
	     FNet   : Tstrings;
	     FNetP  : Tstrings;
	     FEmail : Tstrings;
	     FProxy : TProxyX;
	  published
	     property Net            : TStrings read FNet   write FNet;
	     property NetProtocol    : TStrings read FNetP  write FNetP;
	     property Email_Accounts : TStrings read FEmail write FEmail;
	     property ProxyServer    : TProxyX  read FProxy write FProxy;
	  end;

	  THardwareX=class(TPersistent)
	  private
	     MotherB      : TMotherboardX;
	     FCpu         : TCpuX;
	     FDisplay     : TDisplayX;
	     FInput       : TInputX;
	     FMulti       : TMultiMediaX;
	     FPrinter     : TPrinterX;
	     FKernel      : TKernelX; // AW - 13/07/1999
	     FVFAT        : TVFATX; // AW - 13/07/1999
	     FInt         : TInterfacesX; // AW - 14/07/1999
	     FController  : TControllerX;
	     FNetwork     : TNetX;
	  published
	     property Motherboard : TMotherboardX read MotherB     write MotherB;
	     property CPU         : TCpuX         read FCpu        write FCpu;
	     property Display     : TDisplayX     read FDisplay    write FDisplay;
	     property Input       : TInputX       read FInput      write FInput;
	     property MultiMedia  : TMultiMediaX  read FMulti      write FMulti;
	     property Printer     : TPrinterX     read FPrinter    write FPrinter;
	     property Kernel      : TKernelX      read FKernel     write FKernel; // AW - 13/07/1999
	     property VFAT        : TVFATX        read FVFAT       write FVFAT; // AW - 13/07/1999
	     property Interfaces  : TInterfacesX  read FInt        write FInt; // AW - 14/07/1999
	     property Controller  : TControllerX  read FController write FController; // AW - 14/07/1999
	     property Network     : TNetX         read FNetwork    write FNetwork; // AW - 14/07/1999
	  end;
 //=== End THARDWAREX Classes ==============================================


 //=== TMemory Classes =====================================================
	  TMemoryX=class(TPersistent)
    	  private
 	    FTotalV, FTotalP, FTotalS, FAvailP, FAvailS, FAvailV:longint;
            {$IFDEF WIN98}
	      function GetSystemRes : Byte;
	      function GetGDIRes    : Byte;
              function GetUSERRes   : Byte;
              procedure SetNone(Value: Byte);
            {$ENDIF}
            function GetMemoryStatus(Mode : Byte) : LongInt;
	 published
            {$IFDEF WIN98}
	      property ResourcesSystem          : Byte    read GetSystemRes write SetNone;
      	      property ResourcesGDI             : Byte    read GetGDIRes    write SetNone;
	      property ResourcesUser            : Byte    read GetUserRes   write SetNone;
            {$ENDIF}
	    property PhysicalMemory_Total     : longint read FTotalP      write FTotalP;
	    property PhysicalMemory_Available : longint read FAvailP      write FAvailP;
	    property VirtualMemory_Total      : longint read FTotalV      write FTotalV;
	    property VirtualMemory_Available  : longint read FAvailV      write FAvailV;
	    property SwapMemory_Total         : longint read FTotalS      write FTotalS;
	    property SwapMemory_Available     : longint read FAvailS      write FAvailS;
	 end;
 //=== End TMemory Classes ==================================================


 //=== TSystemX Classes =====================================================
 	 TSystemPathX=class(TPersistent)
	 private
	    FWallPaperDir, FSystemRoot, FCommonFilesDir, FConfigPath                         : string;
	    FDevicePath, FMediaPath, FOtherDevicePath, FProgramFilesPath                     : string;
	    FSendTo, FDesktop, FTemplates, FAppData, FStartMenu, FPrograms, FStartup, FFonts : string;
	    FRecent, FFavorites, FNetHood, FPersonal, FCache, FCookies, FHistory             : string;
	 published
	    property OtherDevicePath    : string read FOtherDevicePath  write FOtherDevicePath;
	    property MediaPath          : string read FMediaPath        write FMediaPath;
	    property DevicePath         : string read FDevicePath       write FDevicePath;
	    property ConfigPath         : string read FConfigPath       write FConfigPath;
	    property CommonFilesDir     : string read FCommonFilesDir   write FCommonFilesDir;
	    property SystemRoot         : string read FSystemRoot       write FSystemRoot;
	    property ProgramFilesPath   : string read FProgramFilesPath write FProgramFilesPath;
	    property WallPaperDir       : string read FWallPaperDir     write FWallPaperDir;
	    property Desktop            : string read FDesktop          write FDesktop;
	    property Templates          : string read FTemplates        write FTemplates;
	    property AppData            : string read FAppData          write FAppData;
	    property Start_Menu         : string read FStartMenu        write FStartMenu;
	    property Programs           : string read FPrograms         write FPrograms;
	    property StartUp            : string read FStartup          write FStartup;
	    property Fonts              : string read FFonts            write FFonts;
	    property SendTo             : string read FSendTo           write FSendTo;
	    property Recent             : string read FRecent           write FRecent;
	    property Favorites          : string read FFavorites        write FFavorites;
	    property NetHood            : string read FNetHood          write FNethood;
	    property Personal           : string read FPersonal         write FPersonal;
	    property Cache              : string read FCache            write FCache;
	    property Cookies            : string read FCookies          write FCookies;
	    property History            : string read FHistory          write FHistory;
	 end;

         TFontsX=class(TPersistent)
         private
           FFontname, FFontType, FFontFile : TStrings;
         published
           property Fontname : TStrings read FFontname write FFontname; // AW - 22/07/1999
           property FontType : TStrings read FFontType write FFontType; // AW - 22/07/1999
           property FontFile : TStrings read FFontFile write FFontFile; // AW - 22/07/1999
         end;

         TSystemMetricsX=class(TPersistent)
         private
           FBootInfo, FBorderSizes, FCursorSize, FDefaultIconSize, FDialogFrameSize,
           FSizableFrameSize, FThickFrameSize, FUsableFullScreen, FMenuDropAlignment : String;
           FPenwindowsSupport : Boolean;
         published
           property BootInfo          : String  read FBootInfo          write FBootInfo; // AW - 22/07/1999
           property BorderSizes       : String  read FBorderSizes       write FBorderSizes; // AW - 22/07/1999
           property CursorSize        : String  read FCursorSize        write FCursorSize; // AW - 22/07/1999
           property DefaultIconSize   : String  read FDefaultIconSize   write FDefaultIconSize; // AW - 22/07/1999
           property DialogFrameSize   : String  read FDialogFrameSize   write FDialogFrameSize; // AW - 22/07/1999
           property SizableFrameSize  : String  read FSizableFrameSize  write FSizableFrameSize; // AW - 22/07/1999
           property ThickFrameSize    : String  read FThickFrameSize    write FThickFrameSize; // AW - 22/07/1999
           property UsableFullScreen  : String  read FUsableFullScreen  write FUsableFullScreen; // AW - 22/07/1999
           property PenwindowsSupport : Boolean read FPenwindowsSupport write FPenwindowsSupport; // AW - 22/07/1999
           property MenuDropAlignment : String  read FMenuDropAlignment write FMenuDropAlignment; // AW - 22/07/1999
         end;

         TEnvironmentX=class(TPersistent)
         private
            FFonts         : TFontsX;
            FSystemMetrics : TSystemMetricsX;
            FEnviron       : TStrings;
         published
            property Environment   : TStrings        read FEnviron       write FEnviron;
            property Fonts         : TFontsX         read FFonts         write FFonts; // AW - 22/07/1999
            property SystemMetrics : TSystemMetricsX read FSystemMetrics write FSystemMetrics; // AW - 22/07/1999
         end;

         TDirectX=class(TPersistent)
         private
            FDirectX, FDMusic : string;
            FD3D, FDPlay      : Tstrings;
         published
	    property Version     : String   read FDirectX write FDirectX;
	    property DirectMusic : String   read FDMusic  write FDMusic;
	    property Direct3D    : TStrings read FD3D     write FD3D;
	    property DirectPlay  : TStrings read FDPlay   write FDPlay;
         end;

	 TOperatingSystemX=class(TPersistent)
	 private
	    FProductName, FProductType, FProductId, FProductKey, FVersion, FVersionNumber:string;
	    FRuntime     : string;
            FEnvironment : TEnvironmentX;
            FUnInstall   : TStrings;
            FDirect      : TDirectX;
	 published
	    property ProductType   : string        read FProductType   write FProductType;
	    property ProductName   : string        read FProductName   write FProductName;
	    property ProductId     : string        read FProductId     write FProductId;
	    property ProductKey    : string        read FProductKey    write FProductKey;
	    property VersionNumber : string        read FVersionNumber write FVersionNumber;
	    property Version       : string        read FVersion       write FVersion;
	    property Runtime       : string        read FRuntime       write FRuntime; // AW - 12/07/1999
            property Environment   : TEnvironmentX read FEnvironment   write FEnvironment; // AW 22/07/1999
            property DirectX       : TDirectX      read FDirect        write FDirect;
	    property UnInstall     : TStrings      read FUnInstall     write FUnInstall;
	 end;

	 TUserX=class (TPersistent)
	 private
	    FWorkgroup, FComputerName, FComment, FRegisteredOwner, FRegisteredOrganization:string;
	 published
	    property Workgroup              : string read FWorkgroup              write FWorkgroup;
	    property ComputerName           : string read FComputerName           write FComputerName;
	    property Comment                : string read FComment                write FComment;
	    property RegisteredOrganization : string read FRegisteredOrganization write FRegisteredOrganization;
	    property RegisteredOwner        : string read FRegisteredOwner        write FRegisteredOwner;
	 end;

	 TSystemX=class(TPersistent)
	 private
	    FSP   : TSystemPathX;
	    FOS   : TOperatingSystemX;
	    FUser : TUserX;
	    FSM_AccessoriesName, FBootCount, FPF_AccessoriesName: string;
	 published
	    property System_Path        : TSystemPathX      read FSP                 write FSP;
	    property Operating_System   : TOperatingSystemX read FOS                 write FOS;
	    property User               : TUserX            read FUser               write FUser;
	    property SM_AccessoriesName : string            read FSM_AccessoriesName write FSM_AccessoriesName;
	    property PF_AccessoriesName : string            read FPF_AccessoriesName write FPF_AccessoriesName;
	    property BootCount          : string            read FBootCount          write FBootCount;
	 end;
 //=== End TSystemX Classes ==================================================

 //=== TSystemInfo2000X Classes ===============================================
 TSystemInfo2000X = class(TComponent)
 private
    FHard  : THardwareX;
    FMem   : TMemoryX;
    FAbout : TAbout2000X;
    FSys   : TSystemX;
    Procedure GetWin98_Infos_Refresh;
 public
    Procedure Refresh;
    Procedure InitSystem;
    procedure ReadFontData; // AW - 22/07/1999
 published
    constructor Create(AOwner:TComponent); override;
    destructor Destroy; override;
    property About          : TAbout2000X read FAbout write FAbout;
    property Hardware       : THardwareX  read Fhard  write Fhard;
    property Memory         : TMemoryX    read FMem   write FMem;
    property System         : TSystemX    read FSys   write FSys;
 end;
 //=== End TSystemInfo2000X Classes ============================================

var
  LReg: TRegistry;
  hInst16: THandle;
  SR: Pointer;

{$M+}
{$F+}
{$I x2000.inc}

implementation

uses X2000About;

Function LoadLibrary16(LibraryName: PChar): THandle; stdcall; external kernel32 index 35;
Function GetProcAddress16(Hinstance: THandle; ProcName: PChar): Pointer; stdcall; external kernel32 index 37;

Procedure FreeLibrary16(HInstance: THandle); stdcall; external kernel32 index 36;
{$IFDEF WIN98}
   Procedure QT_Thunk; cdecl; external kernel32 name 'QT_Thunk';
{$ENDIF}

//= Begin : AW - 12/07/1999 ==================================================================
function GetCPUSpeed: Double;
const
  DelayTime = 500;
var
  TimerHi, TimerLo: DWORD;
  PriorityClass, Priority: Integer;
begin
  PriorityClass := GetPriorityClass(GetCurrentProcess);
  Priority := GetThreadPriority(GetCurrentThread);

  SetPriorityClass(GetCurrentProcess, REALTIME_PRIORITY_CLASS);
  SetThreadPriority(GetCurrentThread, THREAD_PRIORITY_TIME_CRITICAL);

  Sleep(10);
  asm
    dw 310Fh // rdtsc
    mov TimerLo, eax
    mov TimerHi, edx
  end;
  Sleep(DelayTime);
  asm
    dw 310Fh // rdtsc
    sub eax, TimerLo
    sbb edx, TimerHi
    mov TimerLo, eax
    mov TimerHi, edx
  end;

  SetThreadPriority(GetCurrentThread, Priority);
  SetPriorityClass(GetCurrentProcess, PriorityClass);

  Result := TimerLo / (1000.0 * DelayTime);
end;

//=== Begin: AW - 22/07/1999 ==================================================
procedure TSystemInfo2000X.ReadFontData;
var
  Reg    : TRegistry;
  s,
  S_Name,
  S_Type,
  S_File : TStringList;
  s1     : string;
  i, j   : integer;
  a, b   : integer;
begin
  Reg := TRegistry.Create;
  with Reg do begin
    RootKey := Root1;
    OpenKey(WIN98_FONTS_KEY, false);

    s      := TStringList.Create;
    S_Name := TStringList.Create;
    S_Type := TStringList.Create;
    S_File := TStringList.Create;

    Reg.GetValueNames(s);
    j  := s.count;
    for i := 1 to j do begin
      s1 := s[Pred(i)];
      if (Pos('(', s1) > 0) and (Pos(')', s1) > 0) then begin
         a := Pos(')', s1);
         b := Pos('(', s1);
         S_Type.Add(Copy(s1, Succ(b), Pred(a - b)));
         Delete(s1, b, Succ(a -  b));
      end else
         S_Type.Add('???-Font');

      S_Name.Add(s1);
      S_File.Add(Reg.ReadString(s[Pred(i)]));
    end;

    with Fsys.FOS.Environment.Fonts do begin
      FontName.Assign(S_Name);  // AW - 22/07/1999
      FontType.Assign(S_Type);  // AW - 22/07/1999
      FontFile.Assign(S_File);  // AW - 22/07/1999
    end;

    Free;
    s.Free;
    S_Name.Free;
    S_Type.Free;
    S_File.Free;
  end;
end;
//= END ======================================================================================


//= Baldemaier Florian =======================================================================
constructor TSystemInfo2000X.Create(AOwner:TComponent);
begin
  inherited create(AOwner);
  Refresh;
end;
//============================================================================================

//= Baldemaier Florian =======================================================================
destructor TSystemInfo2000X.Destroy;
begin
  inherited Destroy;
end;
//============================================================================================


//= Baldemaier Florian =======================================================================
procedure TSystemInfo2000X.Refresh;
begin
  Inherited;
  InitSystem;
  GetWin98_Infos_Refresh;
end;
//============================================================================================


//= Baldemaier Florian =======================================================================
procedure TSystemInfo2000X.InitSystem;
begin
  inherited;

  //Create
  FHard := THardwareX.Create;
  FMem  := TMemoryX.Create;
  FSys  := TSystemX.Create;

  with FSys do begin
    FSP   := TSystemPathX.create;
    FUser := TUserX.create;
    FOS   := TOperatingSystemX.create;
    with FOS do begin
      Environment               := TEnvironmentX.Create;    // AW - 22/07/1999
      Environment.Fonts         := TFontsX.Create;    // AW - 22/07/1999
      Environment.SystemMetrics := TSystemMetricsX.Create; // AW - 22/07/1999
      Environment.FEnviron      := TStringlist.create;
      DirectX                   := TDirectX.Create;
      FUnInstall                := TStringlist.create;
      DirectX.DirectPlay        := TStringlist.create;
    end;
  end;

  with FHard do begin
    MotherB                := TMotherboardX.create;
    FCpu                   := TCpux.create;
    FDisplay               := TDisplayX.create;
    FDisplay.FCapabilities := TCapabilitiesX.create;
    FInput                 := TInputX.create;
    FMulti                 := TMultiMediaX.create;
    FController            := TControllerX.create;
    FPrinter               := TPrinterX.create;

    Kernel                 := TKernelX.Create;
    VFAT                   := TVFATX.Create;
    Interfaces             := TInterfacesX.Create;
    Interfaces.COM1        := TCOMInterfaceX.Create;
    Interfaces.COM2        := TCOMInterfaceX.Create;
    Interfaces.COM3        := TCOMInterfaceX.Create;
    Interfaces.COM4        := TCOMInterfaceX.Create;
    Input.Mouse            := TMouseX.Create;

    Input.FKeyBoard        := TKeyBoardX.Create;
    FController.System_components := TStringList.Create;
    FController.Controller        := TStringList.create;
    FNetwork                      := TNetX.create;
    with Network do begin
      Net            := TStringList.create;
      NetProtocol    := TStringList.create;
      Email_Accounts := TStringList.create;
      FProxy         := TProxyX.create;
    end;

    with FMulti do begin
      FMixer         := '(unavailable)';
      FMediaControll := TStringList.create;
      FAudioCompress := TStringList.create;
      FImageCompress := TStringList.create;
      FWave          := TStringList.create;
      FMidi          := TStringList.create;
    end;

    //Init
    with Kernel do begin
     CPUUsage                   :=0;
     NumberOfThreads            :=0;
     NumberOfDOSVirtualMachines :=0;
    end;

    with VFAT do begin
     BytesReadPerSecond    :=0;
     BytesWrittenPerSecond :=0;
     Dirty                 :=0;
     ReadsPerSecond        :=0;
     WritesPerSecond       :=0;
    end;

    with Interfaces do begin
     COM1.Available  :=false;
     COM2.Available  :=false;           // AW - 14/07/1999
     COM3.Available  :=false;           // AW - 14/07/1999
     COM4.Available  :=false;           // AW - 14/07/1999
     LPT1_Available  :=false;           // AW - 14/07/1999
     LPT2_Available  :=false;           // AW - 14/07/1999
     LPT3_Available  :=false;           // AW - 14/07/1999
     LPT4_Available  :=false;           // AW - 14/07/1999
     COM1.Baudrate   :='(unavailable)'; // AW - 14/07/1999
     COM2.Baudrate   :='(unavailable)'; // AW - 14/07/1999
     COM3.Baudrate   :='(unavailable)'; // AW - 14/07/1999
     COM4.Baudrate   :='(unavailable)'; // AW - 14/07/1999
     COM1.MaxOutputBuffer :=0;          // AW - 14/07/1999
     COM2.MaxOutputBuffer :=0;          // AW - 14/07/1999
     COM3.MaxOutputBuffer :=0;          // AW - 14/07/1999
     COM4.MaxOutputBuffer :=0;          // AW - 14/07/1999
     COM1.MaxInputBuffer :=0;           // AW - 14/07/1999
     COM2.MaxInputBuffer :=0;           // AW - 14/07/1999
     COM3.MaxInputBuffer :=0;           // AW - 14/07/1999
     COM4.MaxInputBuffer :=0;           // AW - 14/07/1999
     COM1.PacketLength :=0;             // AW - 14/07/1999
     COM2.PacketLength :=0;             // AW - 14/07/1999
     COM3.PacketLength :=0;             // AW - 14/07/1999
     COM4.PacketLength :=0;             // AW - 14/07/1999
     COM1.PacketVersion :=0;            // AW - 14/07/1999
     COM2.PacketVersion :=0;            // AW - 14/07/1999
     COM3.PacketVersion :=0;            // AW - 14/07/1999
     COM4.PacketVersion :=0;            // AW - 14/07/1999
    end;

    with FNetWork.ProxyServer do begin
     FFTP           :='(unavailable)';
     FGOPHER        :='(unavailable)';
     FHTTP          :='(unavailable)';
     FHTTPS         :='(unavailable)';
     FSOCKS         :='(unavailable)';
     FProxyOverride :='(unavailable)';
    end;

    with Motherboard do begin
     BiosDate    :='(unavailable)';
     BiosName    :='(unavailable)';
     BiosVer     :='(unavailable)';
     BusType     :='(unavailable)';     // AW - 12/07/1999
     MachineType :='(unavailable)';     // AW - 12/07/1999
    end;

    with Cpu do begin
     Processor          :='(unavailable)';
     ProcessorSpeed     :='(unavailable)'; // AW - 12/07/1999
     ProcessorVendor    :='(unavailable)';
     ProcessorMMX       :='(unavailable)';
     NumberOfProcessors :='(unavailable)'; // AW - 12/07/1999
     ActiveProcessor    :='(unavailable)'; // AW - 12/07/1999
     CoProcessor        :='(unavailable)';
    end;

    with Display do begin
     DisplayDPI                 :='(unavailable)';
     DisplayResolution          :='(unavailable)';
     DisplayBPP                 :='(unavailable)';
     Display_Driver_Description :='(unavailable)';
     Display_Driver             :='(unavailable)';
     Monitor                    :='(unavailable)';
     Modes                      :=TStringList.Create; // AW - 15/07/1999
    end;

    with Display.Capabilities do begin
     FRasterCapabilities    :=tstringlist.create;
     FCurveCapabilities     :=tstringlist.create;
     FLineCapabilities      :=tstringlist.create;
     FPolygonalCapabilities :=tstringlist.create;
     FTextCapabilities      :=tstringlist.create;
    end;

    MultiMedia.Media        := TStringList.Create;
    MultiMedia.Modem        :='(unavailable)';
    Printer.FPrinter        :='(unavailable)';
    Printer.FPort           :='(unavailable)';
    with Input.Mouse do begin
      Available           :=false;
      Description         :='(unavailable)';
      NumberOfButtons     :=0;
      ButtonsExchanged    :=false;
      MouseWheelAvailable := false;
      FDoubleClickSpeed   :='(unavailable)';
      MouseSpeed          :='(unavailable)';
    end;
    with Input.KeyBoard do begin
      Keyboard             :='(unavailable)';
      NumberOfFunctionKeys :='(unavailable)';
    end;
  end;

  with Fmem do begin
    PhysicalMemory_Total     :=0;
    PhysicalMemory_Available :=0;
    SwapMemory_Total         :=0;
    SwapMemory_Available     :=0;
    VirtualMemory_Total      :=0;
    VirtualMemory_Available  :=0;
  end;

  with FSys do begin
    with User do begin
      Workgroup              :='(unavailable)';
      ComputerName           :='(unavailable)';
      Comment                :='(unavailable)';
      RegisteredOwner        :='(unavailable)';
      RegisteredOrganization :='(unavailable)';
    end;
    BootCount                :='(unavailable)';
    PF_AccessoriesName       :='(unavailable)';
    SM_AccessoriesName       :='(unavailable)';

    with Fsp do begin
      WallPaperDir           :='(unavailable)';
      CommonFilesDir         :='(unavailable)';
      ConfigPath             :='(unavailable)';
      DevicePath             :='(unavailable)';
      MediaPath              :='(unavailable)';
      OtherDevicePath        :='(unavailable)';
      SystemRoot             :='(unavailable)';
      ProgramFilesPath       :='(unavailable)';
      Desktop                :='(unavailable)';
      Templates              :='(unavailable)';
      AppData                :='(unavailable)';
      Start_Menu             :='(unavailable)';
      Programs               :='(unavailable)';
      Startup                :='(unavailable)';
      Fonts                  :='(unavailable)';
      SendTo                 :='(unavailable)';
      Recent                 :='(unavailable)';
      Favorites              :='(unavailable)';
      NetHood                :='(unavailable)';
      Personal               :='(unavailable)';
      Cache                  :='(unavailable)';
      Cookies                :='(unavailable)';
      History                :='(unavailable)';
    end;

    with FOS do begin
      with FDirect do begin
        Version              :='(unavailable)';
        DirectMusic          :='(unavailable)';
        Direct3D             :=TStringlist.create;
      end;
      ProductName            :='(unavailable)';
      ProductType            :='(unavailable)';
      ProductId              :='(unavailable)';
      ProductKey             :='(unavailable)';
      Version                :='(unavailable)';
      VersionNumber          :='(unavailable)';
      Runtime                :='(unavailable)';     // AW - 12/07/1999
      with Environment do begin
         with Fonts do begin
            FontName := TStringList.Create;  // AW - 22/07/1999
            FontType := TStringList.Create;  // AW - 22/07/1999
            FontFile := TStringList.Create;  // AW - 22/07/1999
         end;
         with SystemMetrics do begin
            BootInfo          :='(unavailable)';     // AW - 22/07/1999
            BorderSizes       :='(unavailable)';     // AW - 22/07/1999
            CursorSize        :='(unavailable)';     // AW - 22/07/1999
            DefaultIconSize   :='(unavailable)';     // AW - 22/07/1999
            DialogFrameSize   :='(unavailable)';     // AW - 22/07/1999
            SizableFrameSize  :='(unavailable)';     // AW - 22/07/1999
            ThickFrameSize    :='(unavailable)';     // AW - 22/07/1999
            UsableFullScreen  :='(unavailable)';     // AW - 22/07/1999
            PenwindowsSupport :=false;               // AW - 22/07/1999
            MenuDropAlignment :='(unavailable)';     // AW - 22/07/1999
         end;
      end;
    end;
  end;
end;
//============================================================================================


{$IFDEF WIN98}
function GetFreeSysRes(SysRes: Word): Word;
var
  Thunks: Array[0..$20] of Word;
begin
  Thunks[0] := hInst16;
  hInst16 := LoadLibrary16('user.exe');
  if hInst16 < 32 then
    raise Exception.Create('Can''t load USER.EXE!');
  FreeLibrary16(hInst16);
  SR := GetProcAddress16(hInst16, 'GetFreeSystemResources');
  if SR = nil then
    raise Exception.Create('Can''t get address of GetFreeSystemResources!');
  asm
    push SysRes
    mov edx, SR
    call QT_Thunk
    mov Result, ax
  end;
end;


function TMemoryX.GetSystemRes: Byte;
begin
  Result := GetFreeSysRes(cSystemRes);
end;

function TMemoryX.GetGDIRes: Byte;
begin
  Result := GetFreeSysRes(cGDIRes);
end;

function TMemoryX.GetUSERRes: Byte;
begin
  Result := GetFreeSysRes(cUSERRes);
end;

procedure TMemoryX.SetNone(Value: Byte);
begin
end;
{$ENDIF}

function TMemoryX.GetMemoryStatus(Mode : Byte) : LongInt;
var
  MemInfo : TMemoryStatus;
begin
  MemInfo.dwLength := Sizeof (MemInfo);
  GlobalMemoryStatus(MemInfo);
  with MemInfo do
    case Mode of
      0 : result := dwTotalPhys div 1024;
      1 : result := dwAvailPhys div 1024;
      2 : result := dwTotalPageFile div 1024;
      3 : result := dwAvailPageFile div 1024;
      4 : result := dwTotalVirtual div 1024;
      5 : result := dwAvailVirtual div 1024;
    end;
end;

procedure TAbout2000X.Edit;
begin
 with TAboutForm.Create(Application) do
  try
    ShowModal;
  finally
    Free;
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

procedure TSystemInfo2000X.GetWin98_Infos_Refresh;
  // Begin : AW - 14/07/1999
  procedure SetCOMProps(Cmp : TCOMInterfaceX; CP : TCOMMPROP);
  var
     s : String;
  begin
     case CP.dwMaxBaud of
       BAUD_075    : s := '75 bps';
       BAUD_110    : s := '110 bps';
       BAUD_134_5  : s := '134,5 bps';
       BAUD_150    : s := '150 bps';
       BAUD_300    : s := '300 bps';
       BAUD_600    : s := '600 bps';
       BAUD_1200   : s := '1200 bps';
       BAUD_1800   : s := '1800 bps';
       BAUD_2400   : s := '2400 bps';
       BAUD_4800   : s := '4800 bps';
       BAUD_7200   : s := '7200 bps';
       BAUD_9600   : s := '9600 bps';
       BAUD_14400  : s := '14400 bps';
       BAUD_19200  : s := '19200 bps';
       BAUD_38400  : s := '38400 bps';
       BAUD_56K    : s := '56K bps';
       BAUD_57600  : s := '57600 bps';
       BAUD_115200 : s := '115200 bps';
       BAUD_128K   : s := '128K bps';
       BAUD_USER   : s := 'user-specified';
       else          s := 'unknown/not defined: ' + IntToStr(CP.dwMaxBaud);
     end;
     Cmp.Baudrate := s;
     Cmp.MaxOutputBuffer := CP.dwMaxTxQueue;
     Cmp.MaxInputBuffer := CP.dwMaxRxQueue;
     Cmp.PacketLength := CP.wPacketLength;
     Cmp.PacketVersion := CP.wPacketVersion;
  end;
  // End : AW - 12/07/1999
  // Begin : AW - 14/08/1999
  procedure GetListStuff(Reg : TRegistry; Dest : TStrings; Key, Val : String);
  var
    i : Integer;
    Buff : TStrings;
  begin
    Buff := TStringlist.create;
    with Reg do begin
      OpenKey(Key, False);
      GetKeyNames(Buff);
      CloseKey;

      if Buff.count>0 then begin
        for i:=0 to Buff.Count-1 do begin
          OpenKey(Key+'\'+Buff.Strings[i], False);
          if ReadString(Val)<>'' then begin
            Dest.Add(ReadString(Val));
          end;
          CloseKey;
        end;
      end;
    end;
    Buff.Free;
  end;
  procedure DCs_LC(Ndx : Integer; Msg : String);
  var
    dc : HDC;
  begin
    dc := GetDC(0);
    if (GetDeviceCaps(dc, LINECAPS) and Ndx)=Ndx then
        FHard.Display.Capabilities.FLineCapabilities.Add(Msg);
    ReleaseDC(0, dc);
  end;
  procedure DCs_RC(Ndx : Integer; Msg : String);
  var
    dc : HDC;
  begin
    dc := GetDC(0);
    if (GetDeviceCaps(dc, RASTERCAPS) and Ndx)=Ndx then
        FHard.Display.Capabilities.FRasterCapabilities.Add(Msg);
    ReleaseDC(0, dc);
  end;
  procedure DCs_CC(Ndx : Integer; Msg : String);
  var
    dc : HDC;
  begin
    dc := GetDC(0);
    if (GetDeviceCaps(dc, CURVECAPS) and Ndx)=Ndx then
        FHard.Display.Capabilities.FCurveCapabilities.Add(Msg);
    ReleaseDC(0, dc);
  end;
  procedure DCs_PC(Ndx : Integer; Msg : String);
  var
    dc : HDC;
  begin
    dc := GetDC(0);
    if (GetDeviceCaps(dc, POLYGONALCAPS) and Ndx)=Ndx then
        FHard.Display.Capabilities.FPolygonalCapabilities.Add(Msg);
    ReleaseDC(0, dc);
  end;
  procedure DCs_TC(Ndx : Integer; Msg : String);
  var
    dc : HDC;
  begin
    dc := GetDC(0);
    if (GetDeviceCaps(dc, TEXTCAPS) and Ndx)=Ndx then
        FHard.Display.Capabilities.FTextCapabilities.Add(Msg);
    ReleaseDC(0, dc);
  end;
  // End : AW - 14/08/1999
const MAX_PATH=144;
var
  SI : TSystemInfo; // AW - 12/07/1999
  lData : Longint; // AW - 13/07/1999
  COMM  : pChar; // AW - 14/07/1999
  comms : string; // AW - 14/07/1999
  h     : THandle; // AW - 14/07/1999
  c     : integer; // AW - 14/07/1999
  CP    : TCOMMPROP; // AW - 14/07/1999
  CC    : TCOMMCONFIG; // AW - 14/07/1999
  devmode : TDEVMODE; // AW - 15/07/1999
  k, i  : integer;
  EmailTemp, Temp, Temp2, Provider, DrvDate, VXD, AccountName, r:string;
  POP3Server, POP3User, SMTPServer, SMTPName, SMTPEmailAdress, ProxyTemp:string;
  Emails, MediaTemp : TStrings;
  pDevice :pchar;
  pDriver :pchar;
  pPort   :pchar;
  hDMode  :THandle;
  p:pchar;
  dc : HDC;

Label
 2;

begin

  //Init

  LReg:= TRegistry.Create;
  LReg.RootKey := Root1;
  LReg.LazyWrite := false;

  Emails:=TStringlist.create;
  MediaTemp := TStringList.Create;
  dc := GetDC(0);

  //Reset
  FHard.Controller.Controller.clear;
  FHard.Display.Modes.clear;
  FHard.MultiMedia.Media.Clear;
  FHard.Controller.System_Components.Clear;
  FHard.Network.Net.Clear;
  FHard.Network.NetProtocol.Clear;
  FHard.Network.Email_Accounts.Clear;
  FHard.MultiMedia.Media_Controll.Clear;
  FHard.MultiMedia.Audio_Compress.Clear;
  FHard.MultiMedia.Image_Compress.Clear;
  FHard.MultiMedia.Wave.Clear;
  FHard.MultiMedia.Midi.Clear;
  FSys.FOS.FDirect.Direct3D.Clear;
  Emails.Clear;
  MediaTemp.Clear;

  //Hardware
  with Lreg do begin
   with Fhard.Motherboard do begin
     OpenKey(WIN98_Bios_Key, False);
     if ReadString('BIOSDate')<>'' then          BiosDate        := ReadString('BIOSDate');
     if ReadString('BIOSName')<>'' then          BiosName        := ReadString('BIOSName');
     if ReadString('BIOSVersion')<>'' then       BiosVer         := ReadString('BIOSVersion');
     if ReadString('BusType')<>'' then           BusType         := ReadString('BusType');     // AW - 12/07/1999
     if ReadString('MachineType')<>'' then       MachineType     := ReadString('MachineType'); // AW - 12/07/1999
     CloseKey;
   end;
   with FHard.CPU do begin
     OpenKey(WIN98_Cpu_Key, False);
     if ReadString('Identifier')<>'' then         Processor       := ReadString('Identifier');
     if ReadString('VendorIdentifier')<>'' then   ProcessorVendor := ReadString('VendorIdentifier');
     if ReadString('MMXIdentifier')<>'' then      ProcessorMMX    := ReadString('MMXIdentifier');
     CloseKey;

     OpenKey(WIN98_CoProcessor_Key, False);
     if ReadString('DeviceDesc')<>'' then         CoProcessor     := ReadString('DeviceDesc');
     CloseKey;

     ProcessorSpeed := Format('%f MHz', [GetCPUSpeed]);        // AW - 12/07/1999
     GetSystemInfo(SI);                                        // AW - 12/07/1999
     NumberOfProcessors := IntToStr(SI.dwNumberOfProcessors);  // AW - 12/07/1999
     ActiveProcessor    := IntToStr(SI.dwActiveProcessorMask); // AW - 12/07/1999
     ProcessorRevision  := IntToStr(SI.wProcessorRevision);    // AW - 13/07/1999

   end;
   with FHard.Controller do begin
     OpenKey(WIN98_IDE_Key2, False);
     if ReadString('PortDriver')<>'' then  Temp2:=ReadString('PortDriver') else Temp2:='no portdriver';
     CloseKey;

     OpenKey(WIN98_IDE_Key1, False);
     if ReadString('DriverDate')<>'' then  Temp:='('+ReadString('DriverDate')+', ' else temp:='(no Date, ';
     if ReadString('DriverDesc')<>'' then  Controller.Add('IDE Controller: '+temp+temp2+')');
     CloseKey;

     OpenKey(WIN98_SCSI_Key, False);
     if ReadString('DriverDate')<>'' then temp:='('+ReadString('DriverDate')+', ' else temp:='(no Date, ';
     if ReadString('PortDriver')<>'' then temp:=temp+ReadString('PortDriver')+')' else temp:=temp+', no portdriver)';
     if ReadString('DriverDesc')<>'' then Controller.Add('SCSI Controller: '+ReadString('DriverDesc')+temp);
     CloseKey;

     OpenKey(WIN98_USB_Key, False);
     if ReadString('DriverDate')<>'' then temp:='('+ReadString('DriverDate')+', ' else temp:='(no Date, ';
     if ReadString('NTMPDriver')<>'' then temp:=temp+ReadString('NTMPDriver')+')' else temp:=temp+', no portdriver)';
     if ReadString('DriverDesc')<>'' then Controller.Add('USB Controller: '+ReadString('DriverDesc')+temp);
     CloseKey;
   end;
   with Fhard.Display do begin
    OpenKey(WIN98_Display_Key1, False);
    if ReadString('DPILogicalX')<>'' then          DisplayDPI       := ReadString('DPILogicalX');
    if ReadString('Resolution')<>'' then           DisplayResolution:= ReadString('Resolution');
    if ReadString('BitsPerPixel')<>'' then         DisplayBPP       := ReadString('BitsPerPixel');
    CloseKey;

    OpenKey(WIN98_Display_Key2, False);
    if ReadString('DriverDesc')<>'' then           Display_Driver_Description:= ReadString('DriverDesc');
    CloseKey;

    OpenKey(WIN98_Display_Key2+'\Default', False);
    if ReadString('Drv')<>'' then                  Display_Driver:= ReadString('Drv');
    CloseKey;

    OpenKey(WIN98_Monitor_Key, False);
    if ReadString('DriverDesc')<>'' then      Monitor:= ReadString('DriverDesc');
    CloseKey;

    devmode.dmSize        := SizeOf(TDEVMODE);
    devmode.dmDriverExtra := 0;
    for i := 0 to 25 do
      if EnumDisplaySettings(nil, i, devmode) then
        Modes.Add(IntToStr(devmode.dmPelsWidth) + ' x ' + IntToStr(devmode.dmPelsHeight) + ' - ' + IntToStr(devmode.dmBitsPerPel) + 'Bit');
   end;

   for i:= 0 to 30 do
   begin
      r:=inttostr(i);
      if i<10 then r:='0'+r;
      if KeyExists(WIN98_Media_key1+'\00'+r) then
      begin
        OpenKey(WIN98_Media_key1+'\00'+r, False);
        if ReadString('DriverDesc')<>'' then
        begin
          FHard.MultiMedia.Media.Add(ReadString('DriverDesc'));
        end;
        CloseKey;
      end;
   end;
   if Fhard.MultiMedia.Media.count=0 then FHard.MultiMedia.Media.Add('(unavailable)');

   for i:=0 to 30 do
   begin
     r:=inttostr(i);
     if i<10 then r:='0'+r;
      if KeyExists(WIN98_System_key+'\00'+r) then
      begin
        OpenKey(WIN98_System_key+'\00'+r, False);
        if ReadString('DriverDesc')<>'' then
        begin
          Provider:=ReadString('ProviderName');
          DrvDate:=ReadString('DriverDate');
          VXD:=ReadString('DevLoader');
          if Provider='' then Provider:='no Providername';
          if DrvDate='' then DrvDate:='??.??.????';
          if VXD='' then VXD:='no VXD';
          FHard.Controller.System_components.Add(ReadString('DriverDesc')+'  ('+Provider+', '+DrvDate+', '+VXD+')');
        end;
        CloseKey;
      end;
   end;
   if Fhard.Controller.System_Components.count=0 then FHard.Controller.System_Components.Add('(unavailable)');

   for i:=0 to 30 do
   begin
     r:=inttostr(i);
     if i<10 then r:='0'+r;
      if KeyExists(WIN98_Net_Key1+'\00'+r) then
      begin
        OpenKey(WIN98_Net_Key1+'\00'+r, False);
        if ReadString('DriverDesc')<>'' then
        begin
          Provider:=ReadString('ProviderName');
          DrvDate:=ReadString('DriverDate');
          VXD:=ReadString('DeviceVxds');
          if Provider='' then Provider:='no Providername';
          if DrvDate='' then DrvDate:='??.??.????';
          if VXD='' then VXD:='no VXD';
          FHard.Network.Net.Add(ReadString('DriverDesc')+'  ('+Provider+', '+DrvDate+', '+VXD+')');
        end;
         CloseKey;
     end;
   end;

   for i:=0 to 30 do
   begin
     r:=inttostr(i);
     if i<10 then r:='0'+r;
      if KeyExists(WIN98_Net_Key2+'\00'+r) then
      begin
        OpenKey(WIN98_Net_Key2+'\00'+r, False);
        if ReadString('DriverDesc')<>'' then
        begin
          Provider:=ReadString('ProviderName');
          DrvDate:=ReadString('DriverDate');
          VXD:=ReadString('DeviceVxds');
          if Provider='' then Provider:='no Providername';
          if DrvDate='' then DrvDate:='??.??.????';
          if VXD='' then VXD:='no VXD';
          FHard.Network.Net.Add(ReadString('DriverDesc')+'  ('+Provider+', '+DrvDate+', '+VXD+')');
        end;
        CloseKey;
      end;
   end;

   for i:=0 to 30 do
   begin
     r:=inttostr(i);
     if i<10 then r:='0'+r;
      if KeyExists(WIN98_Net_Key3+'\00'+r) then
      begin
        OpenKey(WIN98_Net_Key3+'\00'+r, False);
        if ReadString('DriverDesc')<>'' then
        begin
          Provider:=ReadString('ProviderName');
          DrvDate:=ReadString('DriverDate');
          VXD:=ReadString('DeviceVxds');
          if Provider='' then Provider:='no Providername';
          if DrvDate='' then DrvDate:='??.??.????';
          if VXD='' then VXD:='no VXD';
          FHard.Network.Net.Add(ReadString('DriverDesc')+'  ('+Provider+', '+DrvDate+', '+VXD+')');
         end;
        CloseKey;
      end;
   end;
   if Fhard.NetWork.Net.count=0 then FHard.NetWork.Net.Add('(unavailable)');

   for i:=0 to 30 do
   begin
     r:=inttostr(i);
     if i<10 then r:='0'+r;
      if KeyExists(WIN98_NetProtocol_key+'\00'+r) then
      begin
        OpenKey(WIN98_NetProtocol_key+'\00'+r, False);
        if ReadString('DriverDesc')<>'' then
        begin
          Provider:=ReadString('ProviderName');
          DrvDate:=ReadString('DriverDate');
          VXD:=ReadString('DeviceVxds');
          if Provider='' then Provider:='no Providername';
          if DrvDate='' then DrvDate:='??.??.????';
          if VXD='' then VXD:='no VXD';
           FHard.Network.NetProtocol.Add(ReadString('DriverDesc')+'  ('+Provider+', '+DrvDate+', '+VXD+')');
        end;
        CloseKey;
      end;
   end;
   if Fhard.NetWork.NetProtocol.count=0 then FHard.NetWork.NetProtocol.Add('(unavailable)');

   RootKey := Root2;
   LazyWrite := false;

   OpenKey(WIN98_Email_Key,false);
   GetKeyNames(Emails);
   CloseKey;
   if Emails.Count>0 then begin
      for i:=0 to Emails.Count-1 do begin
        OpenKey(WIN98_Email_Key+'\'+Emails.Strings[i], False);
        AccountName:=ReadString('Account Name');
        POP3Server:=ReadString('POP3 Server');
        POP3User:=ReadString('POP3 User Name');
        SMTPServer:=ReadString('SMTP Server');
        SMTPName:=ReadString('SMTP Display Name');
        SMTPEmailAdress:=ReadString('SMTP Email Address');
        if AccountName='' then AccountName:='no Account Name';
        if POP3User='' then POP3User:='no POP3 User';
        if SMTPName='' then SMTPName:='no SMTP Display Name';
        if SMTPEmailAdress='' then SMTPEmailAdress:='no SMTP Email Adress';

        EmailTemp:=AccountName+'; '+POP3Server+'; '+POP3User+'; '+SMTPServer+'; '+SMTPName+'; '+SMTPEmailAdress;
        if POP3Server<>'' then begin
         if SMTPServer<>'' then begin
          FHard.Network.Email_Accounts.Add(EmailTemp);
         end;
        end;
        CloseKey;
      end;
   end;
   if Fhard.NetWork.Email_Accounts.count=0 then FHard.NetWork.Email_Accounts.Add('(unavailable)');

   OpenKey(WIN98_Proxy_Key,false);
   if ReadString('ProxyOverride')<>'' then FHard.Network.ProxyServer.ProxyOverride:=ReadString('ProxyOverride');
   if ReadString('ProxyServer')<>'' then ProxyTemp:=ReadString('ProxyServer')+';';
   if ProxyTemp<>'' then begin
    for i:=0 to length(ProxyTemp) do
    begin
     if uppercase(copy(ProxyTemp,i,4))='FTP=' then begin
       for k:=i to length(ProxyTemp) do
       begin
        if copy(ProxyTemp,k,1)=';' then begin
         FHard.Network.ProxyServer.FTP:=copy(ProxyTemp,4+i,k-4-i);
         goto 2;
        end;
       end;
     end;
     if uppercase(copy(ProxyTemp,i,7))='GOPHER=' then begin
       for k:=i to length(ProxyTemp) do
       begin
        if copy(ProxyTemp,k,1)=';' then begin
         FHard.Network.ProxyServer.GOPHER:=copy(ProxyTemp,7+i,k-7-i);
         goto 2;
        end;
      end;
     end;
     if uppercase(copy(ProxyTemp,i,5))='HTTP=' then begin
       for k:=i to length(ProxyTemp) do
       begin
        if copy(ProxyTemp,k,1)=';' then begin
         FHard.Network.ProxyServer.HTTP:=copy(ProxyTemp,5+i,k-5-i);
         goto 2;
        end;
       end;
     end;
     if uppercase(copy(ProxyTemp,i,6))='HTTPS=' then begin
       for k:=i to length(ProxyTemp) do
       begin
        if copy(ProxyTemp,k,1)=';' then begin
         FHard.Network.ProxyServer.HTTPS:=copy(ProxyTemp,6+i,k-6-i);
         goto 2;
        end;
       end;
     end;
     if uppercase(copy(ProxyTemp,i,6))='SOCKS=' then
     begin
        for k:=i to length(ProxyTemp) do
        begin
         if copy(ProxyTemp,k,1)=';' then begin
          FHard.Network.ProxyServer.SOCKS:=copy(ProxyTemp,6+i,k-6-i);
         goto 2;
        end;
       end;
     end;

2:
    end;
   end;

   CloseKey;
   RootKey := Root1;
   LazyWrite := false;

   OpenKey(WIN98_Modem_Key+'\0000', False);
   if ReadString('DriverDesc')<>'' then FHard.MultiMedia.Modem:=ReadString('DriverDesc');
   CloseKey;

   {$IFDEF WIN98}
   GetMem(pDevice, cchDeviceName);
   GetMem(pDriver, MAX_PATH);
   GetMem(pPort, MAX_PATH);
   Printer.GetPrinter(pDevice, pDriver, pPort, hDMode);
   if pdevice<>'' then FHard.Printer.Printer:=pDevice;
   if pPort<>'' then FHard.Printer.Port:=pPort;

   FREEMem(pDevice, cchDeviceName);
   FREEMem(pDriver, MAX_PATH);
   FREEMem(pPort, MAX_PATH);
   {$ENDIF}
   //=== Begin: AW - 14/07/1999 =================================================
   Comms := 'COMX';
   COMM  := StrAlloc(5);

   // COM1-4
   with FHard.Interfaces do begin
    for c := Ord('1') to Ord('4') do begin
     comms[4] := Chr(c);
     StrPCopy(COMM, comms);
     CC.dwSize := SizeOf(TCOMMCONFIG);
     GetDefaultCommConfig(COMM, CC, CC.dwSize);
     h := CreateFile(COMM, GENERIC_READ or GENERIC_WRITE, 0, nil, OPEN_EXISTING, 0, 0);
     if h <> INVALID_HANDLE_VALUE then begin
       case c of
          Ord('1') : COM1.Available := true;
          Ord('2') : COM2.Available := true;
          Ord('3') : COM3.Available := true;
          Ord('4') : COM4.Available := true;
       end;
       GetCommProperties(h, CP);
       case c of
          Ord('1') : SetCOMProps(COM1, CP);
          Ord('2') : SetCOMProps(COM2, CP);
          Ord('3') : SetCOMProps(COM3, CP);
          Ord('4') : SetCOMProps(COM4, CP);
       end;
     end else begin
       case c of
          Ord('1') : COM1.Available := false;
          Ord('2') : COM2.Available := false;
          Ord('3') : COM3.Available := false;
          Ord('4') : COM4.Available := false;
       end;
     end;
     CloseHandle(h);
    end;

    //LPT1-4
    comms := 'LPTX';
    for c := Ord('1') to ORD('4') do begin
     comms[4] := Chr(c);
     StrPCopy(COMM, comms);
     h := CreateFile(COMM, GENERIC_READ OR GENERIC_WRITE, 0, nil, OPEN_EXISTING, 0, 0);
     if h = INVALID_HANDLE_VALUE then begin
       case c of
          Ord('1') : LPT1_Available := false;
          Ord('2') : LPT2_Available := false;
          Ord('3') : LPT3_Available := false;
          Ord('4') : LPT4_Available := false;
       end;
     end else begin
       case c of
          Ord('1') : LPT1_Available := true;
          Ord('2') : LPT2_Available := true;
          Ord('3') : LPT3_Available := true;
          Ord('4') : LPT4_Available := true;
       end;
     end;
     CloseHandle(h);
    end;
    StrDispose(COMM);
   end;
   //=== End: AW - 14/07/1999 ===================================================


   //=== Begin: AW - 13/07/1999 =================================================
   with TRegistry.Create do
      try
         RootKey := HKEY_DYN_DATA;
         OpenKey('PerfStats\StatData', false);

         with FHard do begin
           with Kernel do begin
             ReadBinaryData('KERNEL\CPUUsage', lData, GetDataSize('KERNEL\CPUUsage'));
             CPUUsage := lData;

             ReadBinaryData('KERNEL\Threads', lData, GetDataSize('KERNEL\Threads'));
             NumberOfThreads := lData;

             ReadBinaryData('KERNEL\VMs', lData, GetDataSize('KERNEL\VMs'));
             NumberOfDOSVirtualMachines := lData;
           end;

           with VFAT do begin
             ReadBinaryData('VFat\BReadsSec', lData, GetDataSize('VFat\BReadsSec'));
             BytesReadPerSecond := lData div 1024;

             ReadBinaryData('VFat\BWritesSec', lData, GetDataSize('VFat\BWritesSec'));
             BytesWrittenPerSecond := lData div 1024;

             ReadBinaryData('VFat\DirtyData', lData, GetDataSize('VFat\DirtyData'));
             Dirty := lData;

             ReadBinaryData('VFat\ReadsSec', lData, GetDataSize('VFat\ReadsSec'));
             ReadsPerSecond := lData;

             ReadBinaryData('VFat\WritesSec', lData, GetDataSize('VFat\WritesSec'));
             WritesPerSecond := lData;
           end;
         end;
      finally
         free;
      end;
   //=== End: AW - 13/07/1999 =================================================

   //=== Begin: AW - 15/07/1999 ===============================================
   with FHard.Input do begin
     with Mouse do begin
       Available := Boolean(GetSystemMetrics(SM_MOUSEPRESENT));
       NumberOfButtons := GetSystemMetrics(SM_CMOUSEBUTTONS);
       ButtonsExchanged := Boolean(GetSystemMetrics(SM_SWAPBUTTON));
       MouseWheelAvailable := Boolean(GetSystemMetrics(SM_MOUSEWHEELPRESENT));

       RootKey := Root2;
       LazyWrite := false;

       OpenKey(WIN98_Mouse_Key2, False);
       if ReadString('DoubleClickSpeed')<>'' then FDoubleClickSpeed:=ReadString('DoubleClickSpeed');
       if ReadString('MouseSpeed')<>'' then MouseSpeed:=ReadString('MouseSpeed');
       CloseKey;

       RootKey := Root1;
       LazyWrite := false;

       OpenKey(WIN98_Mouse_Key1, False);
       if ReadString('DriverDesc')<>'' then Description := ReadString('DriverDesc');
       CloseKey;
     end;

     OpenKey(WIN98_Keyboard_Key, False);
     if ReadString('DriverDesc')<>'' then KeyBoard.Keyboard:= ReadString('DriverDesc');
     CloseKey;

     KeyBoard.NumberOfFunctionKeys := IntToStr(GetKeyboardType(2));
   end;
   //=== End: AW - 15/07/1999 =================================================

   //=== Begin: AW - 22/07/1999 ===============================================
   ReadFontData;

   with FSys.Operating_System.Environment.SystemMetrics do begin
      case GetSystemMetrics(SM_CLEANBOOT) of
        0: BootInfo :='normal';
        1: BootInfo :='abgesichert';
        2: BootInfo :='abgesichert mit Netzwerk';
      end;
      BorderSizes       := 'x: ' + IntToStr(GetSystemMetrics(SM_CXBORDER)) + ', y: ' + IntToStr(GetSystemMetrics(SM_CYBORDER));
      CursorSize        := 'x: ' + IntToStr(GetSystemMetrics(SM_CXCURSOR)) + ', y: ' + IntToStr(GetSystemMetrics(SM_CYCURSOR));
      DefaultIconSize   := 'x: ' + IntToStr(GetSystemMetrics(SM_CXICON))   + ', y: ' + IntToStr(GetSystemMetrics(SM_CYICON));
      DialogFrameSize   := 'x: ' + IntToStr(GetSystemMetrics(SM_CXDLGFRAME))+ ', y: ' +IntToStr(GetSystemMetrics(SM_CYDLGFRAME));
      SizableFrameSize  := 'x: ' + IntToStr(GetSystemMetrics(SM_CXFRAME))  + ', y: ' + IntToStr(GetSystemMetrics(SM_CYFRAME));
      ThickFrameSize    := 'x: ' + IntToStr(GetSystemMetrics(SM_CXSIZEFRAME))+', y: '+ IntToStr(GetSystemMetrics(SM_CYSIZEFRAME));
      UsableFullScreen  := 'x: ' + IntToStr(GetSystemMetrics(SM_CXFULLSCREEN))+', y: '+IntToStr(GetSystemMetrics(SM_CYFULLSCREEN));
      PenwindowsSupport := Boolean(GetSystemMetrics(SM_PENWINDOWS));
      if Boolean(GetSystemMetrics(SM_MENUDROPALIGNMENT)) then
        MenuDropAlignment := 'rightaligned'
      else
        MenuDropAlignment := 'leftaligned (standard)';
   end;
   //=== End: AW - 22/07/1999 =================================================
   with FSys.Operating_System.FEnvironment do begin
     FEnviron.clear;
     p:=GetEnvironmentStrings;
     while p^<>#0 do begin
       FEnviron.add (strpas(p));
       inc(p,LStrLen(p)+1);
     end;
     FreeEnvironmentStrings(p);
   end;

   //Memory
   with FMem do
   begin
    PhysicalMemory_Total     := GetMemoryStatus(0);
    PhysicalMemory_Available := GetMemoryStatus(1);
    SwapMemory_Total         := GetMemoryStatus(2);
    SwapMemory_Available     := GetMemoryStatus(3);
    VirtualMemory_Total      := GetMemoryStatus(4);
    VirtualMemory_Available  := GetMemoryStatus(5);
   end;

   //System
   with FSys.User do
   begin
    OpenKey(WIN98_Name_Key, False);
    if ReadString('Workgroup')<>'' then              Workgroup              := ReadString('Workgroup');
    if ReadString('ComputerName')<>'' then           ComputerName           := ReadString('ComputerName');
    if ReadString('Comment')<>'' then                Comment                := ReadString('Comment');
    CloseKey;

    OpenKey(WIN98_Win_Key, False);
    if ReadString('RegisteredOwner')<>'' then        RegisteredOwner         := ReadString('RegisteredOwner');
    if ReadString('RegisteredOrganization')<>'' then RegisteredOrganization  := ReadString('RegisteredOrganization');
    if ReadString('BootCount')<>'' then              FSys.BootCount          := ReadString('BootCount');
    if ReadString('PF_AccessoriesName')<>'' then     FSys.PF_AccessoriesName := ReadString('PF_AccessoriesName');
    if ReadString('SM_AccessoriesName')<>'' then     FSys.SM_AccessoriesName := ReadString('SM_AccessoriesName');
    CloseKey;
   end;

   //SystemPath
   with FSys.FSP do begin
    OpenKey(WIN98_Win_Key, False);
    if ReadString('WallPaperDir')<>'' then     WallPaperDir      := ReadString('WallPaperDir');
    if ReadString('CommonFilesDir')<>'' then   CommonFilesDir    := ReadString('CommonFilesDir');
    if ReadString('ConfigPath')<>'' then       ConfigPath        := ReadString('ConfigPath');
    if ReadString('DevicePath')<>'' then       DevicePath        := ReadString('DevicePath');
    if ReadString('MediaPath')<>'' then        MediaPath         := ReadString('MediaPath');
    if ReadString('OtherDevicePath')<>'' then  OtherDevicePath   := ReadString('OtherDevicePath');
    if ReadString('SystemRoot')<>'' then       SystemRoot        := ReadString('SystemRoot');
    if ReadString('ProgramFilesPath')<>'' then ProgramFilesPath  := ReadString('ProgramFilesPath');
    CloseKey;

    RootKey := Root2;
    LazyWrite := false;
    OpenKey(WIN98_Path_Key, False);
    if ReadString('Desktop')<>'' then    Desktop    := ReadString('Desktop');
    if ReadString('Templates')<>'' then  Templates  := ReadString('Templates');
    if ReadString('AppData')<>'' then    AppData    := ReadString('AppData');
    if ReadString('Start Menu')<>'' then Start_Menu := ReadString('Start Menu');
    if ReadString('Programs')<>'' then   Programs   := ReadString('Programs');
    if ReadString('Startup')<>'' then    Startup    := ReadString('Startup');
    if ReadString('Fonts')<>'' then      Fonts      := ReadString('Fonts');
    if ReadString('SendTo')<>'' then     SendTo     := ReadString('SendTo');
    if ReadString('Recent')<>'' then     Recent     := ReadString('Recent');
    if ReadString('Favorites')<>'' then  Favorites  := ReadString('Favorites');
    if ReadString('NetHood')<>'' then    NetHood    := ReadString('NetHood');
    if ReadString('Personal')<>'' then   Personal   := ReadString('Personal');
    if ReadString('Cache')<>'' then      Cache      := ReadString('Cache');
    if ReadString('Cookies')<>'' then    Cookies    := ReadString('Cookies');
    if ReadString('History')<>'' then    History    := ReadString('History');
    CloseKey;
   end;

   //OperatingSystem
   with FSys.FOS do begin
    RootKey := Root1;
    LazyWrite := false;
    OpenKey(WIN98_Win_Key, False);
    if ReadString('ProductName')<>'' then   ProductName   := ReadString('ProductName');
    if ReadString('ProductType')<>'' then   ProductType   := ReadString('ProductType');
    if ReadString('ProductId')<>'' then     ProductId     := ReadString('ProductId');
    if ReadString('ProductKey')<>'' then    ProductKey    := ReadString('ProductKey');
    if ReadString('Version')<>'' then       Version       := ReadString('Version');
    if ReadString('VersionNumber')<>'' then VersionNumber := ReadString('VersionNumber');
    CloseKey;
    i := GetTickCount; // AW - 12/07/1999
    Runtime := (IntToStr(i div (60 * 60 * 1000)) + ' hour(s) and ' + IntToStr((i div (60 * 1000)) - (i div (60 * 60 * 1000)) * 60) + ' minute(s)'); // AW - 12/07/1999

    OpenKey(WIN98_DirectX_Key, False);
    if ReadString('Version')<>'' then DirectX.Version        := ReadString('Version');
    CloseKey;

    OpenKey(WIN98_DirectMusic_Key+'\Defaults', False);
    if ReadString('DefaultOutputPort')<>'' then temp:=ReadString('DefaultOutputPort');
    CloseKey;

    if KeyExists(WIN98_DirectMusic_Key+'\SoftwareSynths\'+temp) then begin
       OpenKey(WIN98_DirectMusic_Key+'\SoftwareSynths\'+temp, False);
       if ReadString('Description')<>'' then FSys.Fos.DirectX.DirectMusic:= ReadString('Description');
       Closekey;
    end;

    GetListStuff(Lreg, FSys.FOS.DirectX.Direct3D,   WIN98_Direct3D_Key,   'Description');
    GetListStuff(Lreg, FSys.FOS.DirectX.DirectPlay, WIN98_DirectPlay_Key, 'DescriptionA');
    GetListStuff(Lreg, FSys.FOS.FUnInstall,         WIN98_UnInstall_Key,  'DisplayName');

    with FHard.MultiMedia do begin
      GetListStuff(Lreg, Image_Compress, WIN98_Media_Key2+'\ICM', 'Description');
      GetListStuff(Lreg, Media_Controll, WIN98_Media_Key2+'\MCI', 'FriendlyName');
      GetListStuff(Lreg, Wave,           WIN98_Media_Key2+'\WAVE','FriendlyName');
      GetListStuff(Lreg, Midi,           WIN98_Media_Key2+'\MIDI','FriendlyName');
      GetListStuff(Lreg, Audio_Compress, WIN98_Media_Key2+'\ACM', 'Description');
    end;

    MediaTemp.Clear;
    OpenKey(WIN98_Media_Key2+'\MIXER', False);
    GetKeyNames(MediaTemp);
    CloseKey;

    if MediaTemp.count>0 then begin
      OpenKey(WIN98_Media_Key2+'\MIXER\'+MediaTemp.Strings[0], False);
      if ReadString('FriendlyName')<>'' then FHard.MultiMedia.Mixer:=ReadString('FriendlyName');
      CloseKey;
    end;

   end;
  end;

  with FHard.Display.Capabilities do begin
    {raster capabilities}
    FRasterCapabilities.Clear;
    DCs_RC(RC_BANDING,      'Requires Banding');
    DCs_RC(RC_BITBLT,       'Can Transer Bitmaps');
    DCs_RC(RC_BITMAP64,     'Supports Bitmaps > 64K');
    DCs_RC(RC_DI_BITMAP,    'Supports SetDIBits and GetDIBits');
    DCs_RC(RC_DIBTODEV,     'Supports SetDIBitsToDevice');
    DCs_RC(RC_FLOODFILL,    'Can Perform Floodfills');
    DCs_RC(RC_GDI20_OUTPUT, 'Supports Windows 2.0 Features');
    DCs_RC(RC_PALETTE,      'Palette Based');
    DCs_RC(RC_SCALING,      'Supports Scaling');
    DCs_RC(RC_STRETCHBLT,   'Supports StretchBlt');
    DCs_RC(RC_STRETCHDIB,   'Supports StretchDIBits');

    {curve capabilities}
    FCurveCapabilities.clear;
    if GetDeviceCaps(dc, CURVECAPS)=CC_NONE then
      FCurveCapabilities.Add('Device Does Not Support Curves')
    else
    begin
      DCs_CC(CC_CIRCLES,    'Supports Cirles');
      DCs_CC(CC_PIE,        'Supports Pie Wedges');
      DCs_CC(CC_CHORD,      'Supports Chords');
      DCs_CC(CC_ELLIPSES,   'Supports Ellipses');
      DCs_CC(CC_WIDE,       'Supports Wide Borders');
      DCs_CC(CC_STYLED,     'Supports Styled Borders');
      DCs_CC(CC_WIDESTYLED, 'Supports Wide And Styled Borders');
      DCs_CC(CC_INTERIORS,  'Supports Interiors');
      DCs_CC(CC_ROUNDRECT,  'Supports Rounded Rectangles');
    end;

    {line capabilities}
    FLineCapabilities.clear;
    if GetDeviceCaps(dc, LINECAPS)=LC_NONE then
      FLineCapabilities.Add('Device Does Not Support Lines')
    else
    begin
      DCs_LC(LC_POLYLINE,   'Supports Polylines');
      DCs_LC(LC_MARKER,     'Supports Markers');
      DCs_LC(LC_POLYMARKER, 'Supports Multiple Markers');
      DCs_LC(LC_WIDE,       'Supports Wide Lines');
      DCs_LC(LC_STYLED,     'Supports Styled Lines');
      DCs_LC(LC_WIDESTYLED, 'Supports Wide And Styled Lines');
      DCs_LC(LC_INTERIORS,  'Supports Interiors');
    end;

    {polygonal capabilities}
    FPolygonalCapabilities.clear;
    if GetDeviceCaps(dc, POLYGONALCAPS)=PC_NONE then
      FPolygonalCapabilities.Add('Device Does Not Support Polygons')
    else
    begin
      DCs_PC(PC_POLYGON,     'Supports Alternate Fill Polygons');
      DCs_PC(PC_RECTANGLE,   'Supports Rectangles');
      DCs_PC(PC_WINDPOLYGON, 'Supports Winding Fill Polygons');
      DCs_PC(PC_SCANLINE,    'Supports Single Scanlines');
      DCs_PC(PC_WIDE,        'Supports Wide Borders');
      DCs_PC(PC_STYLED,      'Supports Styled Borders');
      DCs_PC(PC_WIDESTYLED,  'Supports Wide And Styled Borders');
      DCs_PC(PC_INTERIORS,   'Supports Interiors');
    end;

    {text capabilities}
    FTextCapabilities.clear;
    DCs_TC(TC_OP_CHARACTER, 'Capable of Character Output Precision');
    DCs_TC(TC_OP_STROKE,    'Capable of Stroke Output Precision');
    DCs_TC(TC_CP_STROKE,    'Capable of Stroke Clip Precision');
    DCs_TC(TC_CR_90,        'Supports 90 Degree Character Rotation');
    DCs_TC(TC_CR_ANY,       'Supports Character Rotation to Any Angle');
    DCs_TC(TC_SF_X_YINDEP,  'X And Y Scale Independent');
    DCs_TC(TC_SA_DOUBLE,    'Supports Doubled Character Scaling');
    DCs_TC(TC_SA_INTEGER,   'Supports Integer Multiples Only When Scaling');
    DCs_TC(TC_SA_CONTIN,    'Supports Any Multiples For Exact Character Scaling');
    DCs_TC(TC_EA_DOUBLE,    'Supports Double Weight Characters');
    DCs_TC(TC_IA_ABLE,      'Supports Italics');
    DCs_TC(TC_UA_ABLE,      'Supports Underlines');
    DCs_TC(TC_SO_ABLE,      'Supports Strikeouts');
    DCs_TC(TC_RA_ABLE,      'Supports Raster Fonts');
    DCs_TC(TC_VA_ABLE,      'Supports Vector Fonts');
    DCs_TC(TC_SCROLLBLT,    'Cannot Scroll Using Blts');
   end;
   ReleaseDC(0, dc);

  LReg.Free;
  Emails.Free;
 end;
end.
