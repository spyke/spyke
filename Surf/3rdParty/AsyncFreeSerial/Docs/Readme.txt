AsyncFree library
-----------------
Serial communication and related components


Installation:
-------------
Delphi 5: 
  install AsyncFreeD5.dpk

Delphi 4: 
  install AsyncFreeD4.dpk

Delphi 3:
  install AsyncFreeD3.dpk

Note: The code is developed and primary tested on Delphi 4.03 under
Windows 95 OSR2.


File contents:
--------------
Docs\AsyncFree.rtf	Help file 
Docs\Readme.txt		The file you're probably reading now

AsyncFreeD3.dpk		Delphi 3 package
AsyncFreeD4.dpk		Delphi 4 package
AsyncFreeD5.dpk		Delphi 5 package
AsyncFreeD3.res		Delphi 3 package resource file
AsyncFreeD4.res		Delphi 4 package resource file
AsyncFreeD5.res		Delphi 5 package resource file
AfCircularBuffer.pas	Circular buffer
AfComPort.pas		Serial communication base component
AfComPortCore.pas		Communication core low-level objects
AfDataControls.pas	(not finished yet, do not use)
AfDataDispatcher.pas	Data dispatcher and related components
AfDataTerminal.pas	"Data-aware" terminal 
AfPortControls.pas	Serial port selection visible components
AfRegister.pas		Property editors and its registering
AfSafeSync.pas		VCL synchronization functions
AfUtils.pas			Miscellaneous functions
AfViewers.pas		Terminal and file viewer 
AfRegister.dcr		Component icons
PvDefine.inc		Delphi version definitions


Examples: (\Examples folder)
----------------------------
DataAwareExample		It shows how to make descendant of the standard
				control (TMemo) which represents an AsyncFree 
				"data-aware" component.

FileViewerExample		Example of using TAfFileViewer component. 

LineViewerExample		Example of using TAfLineViewer component. It shows
				how to work with different color modes that allows
				you make syntax highlighting for log text files.

NonsyncEventsExample	Example of using non synchronized events and its
				advantages comparing to normal events.

NonVCLExample		It shows how to use core low-level objects in a 
				small	console application without using components.

SimplePortExample		Example of using base serial communication component
				TAfComPort.

TerminalPortExample	Example of using data dispatcher and "data-aware"
				components. There is also shown how to send a file
				through serial port using TStream class.

AsyncFreeExamplesGroup  The examples group for Delphi 4 and 5 users


Known bugs:
-----------
* TAfComPort.OnOutBufFree event isn't fired properly when there is a huge
  incoming data traffic.
* When TAfTerminal logging mode is lgCreate, the log file doesn't allow
  file read sharing.


Future:
-------
* Improve documentation and examples as well (and my english, of course :-)
* Add "text line received" event to the data dispatcher to allow work with 
  a single line.
* Add user timers. I thought to put them into data dispatcher to make a 
  complex component for working with data. But now I decided to put them 
  to the separate component to make it more flexible and independent of
  having using data dispatcher.
* TAPI support
* RAS encapsulation (???)


Contact:
--------
Delphree - The Open Source Delphi Development Initiative:
  http://delphree.clexpert.com

Initial developer: 
  Petr Vones - petr.v@mujmail.cz

  