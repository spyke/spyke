//----------------------------------------------------------------------------
// TimerEx.hpp - bcbdcc32 generated hdr (DO NOT EDIT) rev: 0
// From: TimerEx.pas
//----------------------------------------------------------------------------
#ifndef TimerExHPP
#define TimerExHPP
//----------------------------------------------------------------------------
#include <Classes.hpp>
#include <Messages.hpp>
#include <SysUtils.hpp>
#include <Windows.hpp>
#include <System.hpp>
namespace Timerex
{
//-- type declarations -------------------------------------------------------
class __declspec(delphiclass) EOutOfMemory;
class __declspec(pascalimplementation) EOutOfMemory : public Sysutils::Exception
{
	typedef Sysutils::Exception inherited;
	
public:
	/* Exception.Create */ __fastcall EOutOfMemory(const System::AnsiString Msg) : Sysutils::Exception(
		Msg) { }
	/* Exception.CreateFmt */ __fastcall EOutOfMemory(const System::AnsiString Msg, const System::TVarRec 
		* Args, const int Args_Size) : Sysutils::Exception(Msg, Args, Args_Size) { }
	/* Exception.CreateRes */ __fastcall EOutOfMemory(int Ident) : Sysutils::Exception(Ident) { }
	/* Exception.CreateResFmt */ __fastcall EOutOfMemory(int Ident, const System::TVarRec * Args, const 
		int Args_Size) : Sysutils::Exception(Ident, Args, Args_Size) { }
	/* Exception.CreateHelp */ __fastcall EOutOfMemory(const System::AnsiString Msg, int AHelpContext) : 
		Sysutils::Exception(Msg, AHelpContext) { }
	/* Exception.CreateFmtHelp */ __fastcall EOutOfMemory(const System::AnsiString Msg, const System::TVarRec 
		* Args, const int Args_Size, int AHelpContext) : Sysutils::Exception(Msg, Args, Args_Size, AHelpContext
		) { }
	/* Exception.CreateResHelp */ __fastcall EOutOfMemory(int Ident, int AHelpContext) : Sysutils::Exception(
		Ident, AHelpContext) { }
	/* Exception.CreateResFmtHelp */ __fastcall EOutOfMemory(int Ident, const System::TVarRec * Args, const 
		int Args_Size, int AHelpContext) : Sysutils::Exception(Ident, Args, Args_Size, AHelpContext) { }
	
public:
	/* TObject.Destroy */ __fastcall virtual ~EOutOfMemory(void) { }
	
};

class __declspec(delphiclass) EOutOfResources;
class __declspec(pascalimplementation) EOutOfResources : public EOutOfMemory
{
	typedef EOutOfMemory inherited;
	
public:
	/* Exception.Create */ __fastcall EOutOfResources(const System::AnsiString Msg) : Timerex::EOutOfMemory(
		Msg) { }
	/* Exception.CreateFmt */ __fastcall EOutOfResources(const System::AnsiString Msg, const System::TVarRec 
		* Args, const int Args_Size) : Timerex::EOutOfMemory(Msg, Args, Args_Size) { }
	/* Exception.CreateRes */ __fastcall EOutOfResources(int Ident) : Timerex::EOutOfMemory(Ident) { }
	/* Exception.CreateResFmt */ __fastcall EOutOfResources(int Ident, const System::TVarRec * Args, const 
		int Args_Size) : Timerex::EOutOfMemory(Ident, Args, Args_Size) { }
	/* Exception.CreateHelp */ __fastcall EOutOfResources(const System::AnsiString Msg, int AHelpContext
		) : Timerex::EOutOfMemory(Msg, AHelpContext) { }
	/* Exception.CreateFmtHelp */ __fastcall EOutOfResources(const System::AnsiString Msg, const System::TVarRec 
		* Args, const int Args_Size, int AHelpContext) : Timerex::EOutOfMemory(Msg, Args, Args_Size, AHelpContext
		) { }
	/* Exception.CreateResHelp */ __fastcall EOutOfResources(int Ident, int AHelpContext) : Timerex::EOutOfMemory(
		Ident, AHelpContext) { }
	/* Exception.CreateResFmtHelp */ __fastcall EOutOfResources(int Ident, const System::TVarRec * Args
		, const int Args_Size, int AHelpContext) : Timerex::EOutOfMemory(Ident, Args, Args_Size, AHelpContext
		) { }
	
public:
	/* TObject.Destroy */ __fastcall virtual ~EOutOfResources(void) { }
	
};

class __declspec(delphiclass) TTimerEx;
class __declspec(pascalimplementation) TTimerEx : public Classes::TComponent
{
	typedef Classes::TComponent inherited;
	
private:
	bool fEnabled;
	Cardinal fInterval;
	Classes::TNotifyEvent fOnTimer;
	bool fThreaded;
	TThreadPriority fThreadPriority;
	bool fThreadSafe;
	Classes::TThread* fTimerThread;
	HWND FWindowHandle;
	void __fastcall SetEnabled(bool p0);
	void __fastcall SetInterval(Cardinal p0);
	void __fastcall SetOnTimer(Classes::TNotifyEvent p0);
	void __fastcall SetThreaded(bool p0);
	void __fastcall SetThreadPriority(Classes::TThreadPriority p0);
	void __fastcall UpdateTimer(void);
	void __fastcall WndProc(Messages::TMessage &TimerEx_);
	
protected:
	virtual void __fastcall Timer(void);
	
public:
	__fastcall virtual TTimerEx(Classes::TComponent* p0);
	__fastcall virtual ~TTimerEx(void);
	
__published:
	__property bool Enabled = {read=fEnabled, write=SetEnabled, default=1};
	__property Cardinal Interval = {read=fInterval, write=SetInterval, default=1000};
	__property bool Threaded = {read=fThreaded, write=SetThreaded, default=0};
	__property Classes::TThreadPriority ThreadPriority = {read=fThreadPriority, write=SetThreadPriority
		, default=3};
	__property bool ThreadSafe = {read=fThreadSafe, write=fThreadSafe, nodefault};
	__property Classes::TNotifyEvent OnTimer = {read=fOnTimer, write=SetOnTimer};
};

//-- var, const, procedure ---------------------------------------------------
#define DEFAULT_INTERVAL (Word)(1000)
#define DEFAULT_THREADED (bool)(0)

}	/* namespace Timerex */
#if !defined(NO_IMPLICIT_NAMESPACE_USE)
using namespace Timerex;
#endif
//-- end unit ----------------------------------------------------------------
#endif	// TimerEx
