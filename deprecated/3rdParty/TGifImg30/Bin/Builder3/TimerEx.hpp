// Borland C++ Builder
// Copyright (c) 1995, 1998 by Borland International
// All rights reserved

// (DO NOT EDIT: machine generated header) 'TimerEx.pas' rev: 3.00

#ifndef TimerExHPP
#define TimerExHPP
#include <Classes.hpp>
#include <Messages.hpp>
#include <Windows.hpp>
#include <SysInit.hpp>
#include <System.hpp>

//-- user supplied -----------------------------------------------------------

namespace Timerex
{
//-- type declarations -------------------------------------------------------
class DELPHICLASS TTimerEx;
class PASCALIMPLEMENTATION TTimerEx : public Classes::TComponent 
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
	void __fastcall SetEnabled(bool aValue);
	void __fastcall SetInterval(Cardinal aValue);
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
#define DEFAULT_PRIORITY (Classes::TThreadPriority)(3)

}	/* namespace Timerex */
#if !defined(NO_IMPLICIT_NAMESPACE_USE)
using namespace Timerex;
#endif
//-- end unit ----------------------------------------------------------------
#endif	// TimerEx
