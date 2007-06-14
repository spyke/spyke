// Borland C++ Builder
// Copyright (c) 1995, 1999 by Borland International
// All rights reserved

// (DO NOT EDIT: machine generated header) 'TimerEx.pas' rev: 4.00

#ifndef TimerExHPP
#define TimerExHPP

#pragma delphiheader begin
#pragma option push -w-
#include <Classes.hpp>	// Pascal unit
#include <Messages.hpp>	// Pascal unit
#include <Windows.hpp>	// Pascal unit
#include <SysInit.hpp>	// Pascal unit
#include <System.hpp>	// Pascal unit

//-- user supplied -----------------------------------------------------------

namespace Timerex
{
//-- type declarations -------------------------------------------------------
class DELPHICLASS TTimerEx;
#pragma pack(push, 4)
class PASCALIMPLEMENTATION TTimerEx : public Classes::TComponent 
{
	typedef Classes::TComponent inherited;
	
private:
	bool fEnabled;
	unsigned fInterval;
	Classes::TNotifyEvent fOnTimer;
	bool fThreaded;
	Classes::TThreadPriority fThreadPriority;
	bool fThreadSafe;
	Classes::TThread* fTimerThread;
	HWND FWindowHandle;
	void __fastcall SetEnabled(bool aValue);
	void __fastcall SetInterval(unsigned aValue);
	void __fastcall SetOnTimer(Classes::TNotifyEvent aValue);
	void __fastcall SetThreaded(bool aValue);
	void __fastcall SetThreadPriority(Classes::TThreadPriority aValue);
	void __fastcall UpdateTimer(void);
	void __fastcall WndProc(Messages::TMessage &aMessage);
	
protected:
	DYNAMIC void __fastcall Timer(void);
	
public:
	__fastcall virtual TTimerEx(Classes::TComponent* aOwner);
	__fastcall virtual ~TTimerEx(void);
	
__published:
	__property bool Enabled = {read=fEnabled, write=SetEnabled, default=1};
	__property unsigned Interval = {read=fInterval, write=SetInterval, default=1000};
	__property bool Threaded = {read=fThreaded, write=SetThreaded, default=0};
	__property Classes::TThreadPriority ThreadPriority = {read=fThreadPriority, write=SetThreadPriority
		, default=3};
	__property bool ThreadSafe = {read=fThreadSafe, write=fThreadSafe, nodefault};
	__property Classes::TNotifyEvent OnTimer = {read=fOnTimer, write=SetOnTimer};
};

#pragma pack(pop)

//-- var, const, procedure ---------------------------------------------------
static const Word DEFAULT_INTERVAL = 0x3e8;
static const bool DEFAULT_THREADED = false;
#define DEFAULT_PRIORITY (Classes::TThreadPriority)(3)

}	/* namespace Timerex */
#if !defined(NO_IMPLICIT_NAMESPACE_USE)
using namespace Timerex;
#endif
#pragma option pop	// -w-

#pragma delphiheader end.
//-- end unit ----------------------------------------------------------------
#endif	// TimerEx
