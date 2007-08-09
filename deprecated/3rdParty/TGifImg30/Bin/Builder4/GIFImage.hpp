// Borland C++ Builder
// Copyright (c) 1995, 1999 by Borland International
// All rights reserved

// (DO NOT EDIT: machine generated header) 'GIFImage.pas' rev: 4.00

#ifndef GIFImageHPP
#define GIFImageHPP

#pragma delphiheader begin
#pragma option push -w-
#include <Menus.hpp>	// Pascal unit
#include <TimerEx.hpp>	// Pascal unit
#include <Controls.hpp>	// Pascal unit
#include <Classes.hpp>	// Pascal unit
#include <Graphics.hpp>	// Pascal unit
#include <SysUtils.hpp>	// Pascal unit
#include <Messages.hpp>	// Pascal unit
#include <Windows.hpp>	// Pascal unit
#include <SysInit.hpp>	// Pascal unit
#include <System.hpp>	// Pascal unit

//-- user supplied -----------------------------------------------------------

namespace Gifimage
{
//-- type declarations -------------------------------------------------------
#pragma option push -b-
enum TGIFVersion { gvGIF87a, gvGIF89a };
#pragma option pop

#pragma pack(push, 1)
struct TRGBColor
{
	Byte Red;
	Byte Green;
	Byte Blue;
} ;
#pragma pack(pop)

typedef TRGBColor TColorTable[256];

typedef TRGBColor *PColorTable;

#pragma option push -b-
enum TDisposalType { dtUndefined, dtDoNothing, dtToBackground, dtToPrevious };
#pragma option pop

#pragma pack(push, 4)
struct TFrameInfo
{
	Graphics::TBitmap* iiImage;
	int iiLeft;
	int iiTop;
	int iiWidth;
	int iiHeight;
	int iiDelay;
	bool iiInterlaced;
	bool iiTransparent;
	Graphics::TColor iiTransparentColor;
	TDisposalType iiDisposalMethod;
	AnsiString iiComment;
} ;
#pragma pack(pop)

#pragma pack(push, 4)
struct TSaveInfo
{
	int siNumFrames;
	TFrameInfo *siFrames[256];
	Word siNumLoops;
	bool siUseGlobalColorTable;
} ;
#pragma pack(pop)

#pragma option push -b-
enum TGIFError { geNone, geCanceled, geInternal, geFileFormat, geTooManyColors, geIndexOutOfBounds, 
	geWindows, geFileNotFound, geResourceNotFound };
#pragma option pop

typedef void __fastcall (__closure *TProgressEvent)(System::TObject* Sender, const int BytesProcessed
	, const int BytesToProcess, int PercentageProcessed, bool &KeepOnProcessing);

class DELPHICLASS TGIFFrame;
class DELPHICLASS TGIFImage;
#pragma pack(push, 4)
class PASCALIMPLEMENTATION TGIFImage : public Controls::TGraphicControl 
{
	typedef Controls::TGraphicControl inherited;
	
private:
	bool FAnimated;
	bool FAnimate;
	bool FAutoSize;
	Graphics::TBitmap* FBitmap;
	bool FCenter;
	int FCurrentFrame;
	Classes::TMemoryStream* FData;
	bool FDoubleBuffered;
	bool FEmpty;
	bool FFirstImageOnly;
	int FImageWidth;
	int FImageHeight;
	bool FInterlaced;
	bool FLoop;
	bool FMouseOnTransparent;
	int FNumFrames;
	int FNumIterations;
	bool FOpaque;
	int FSpeed;
	bool FStretch;
	bool FStretchRatio;
	bool FStretchBigOnly;
	bool FThreaded;
	Classes::TThreadPriority FThreadPriority;
	bool FTile;
	bool FTransparent;
	bool FVisible;
	TGIFVersion FVersion;
	Classes::TNotifyEvent FOnChanging;
	Classes::TNotifyEvent FOnChange;
	TProgressEvent FOnProgress;
	Classes::TNotifyEvent FOnWrapAnimation;
	Graphics::TBitmap* Image;
	Graphics::TBitmap* ImageMask;
	int FBitsPerPixel;
	TRGBColor GlobalColorTable[256];
	TGIFFrame* *Frames[256];
	int CurrentIteration;
	bool Fresh;
	bool FRepainting;
	bool Have2Stretch;
	bool CoversClientArea;
	bool HaveInfoOnly;
	Classes::TMemoryStream* MemStream;
	bool KeepOnDecoding;
	Timerex::TTimerEx* Timer;
	void __fastcall SetThreaded(bool aValue);
	void __fastcall SetThreadPriority(Classes::TThreadPriority aValue);
	void __fastcall SetAnimate(bool aAnimate);
	HIDESBASE void __fastcall SetAutoSize(bool aAutoSize);
	void __fastcall SetCenter(bool aCenter);
	void __fastcall SetFrame(int aFrame);
	void __fastcall SetLoop(bool aLoop);
	void __fastcall SetOpaque(const bool Value);
	void __fastcall SetSpeed(int aSpeed);
	void __fastcall SetStretch(bool aStretch);
	void __fastcall SetStretchRatio(bool aStretchRatio);
	void __fastcall SetTile(bool aTile);
	HIDESBASE void __fastcall SetVisible(bool aVisible);
	System::TObject* __fastcall GetGraphic(void);
	void __fastcall SetGraphic(const System::TObject* Value);
	void __fastcall UpdateTimer(const bool Enabled, const int Interval);
	Graphics::TBitmap* __fastcall GetBitmap(void);
	Windows::TRect __fastcall GetClipRect(void);
	void __fastcall UpdateOpacity(void);
	HIDESBASE MESSAGE void __fastcall WMPaint(Messages::TWMPaint &Msg);
	HIDESBASE MESSAGE void __fastcall WMMouseMove(Messages::TWMMouse &Msg);
	void __fastcall SetStretchBigOnly(const bool Value);
	
protected:
	virtual void __fastcall AssignTo(Classes::TPersistent* Dest);
	virtual void __fastcall Loaded(void);
	virtual void __fastcall DefineProperties(Classes::TFiler* Filer);
	virtual void __fastcall ReadData(Classes::TStream* Stream);
	virtual void __fastcall WriteData(Classes::TStream* Stream);
	DYNAMIC HPALETTE __fastcall GetPalette(void);
	virtual bool __fastcall ReadGIF(void);
	virtual void __fastcall Changing(void);
	virtual void __fastcall Change(void);
	virtual void __fastcall DoProgress(void);
	void __fastcall LocalClear(void);
	void __fastcall SetupBitmap(void);
	void __fastcall RemoveBitmap(int aFrame);
	void __fastcall PaintImage(HDC DC);
	virtual void __fastcall Paint(void);
	void __fastcall OnTrigger(System::TObject* Sender);
	
public:
	TGIFError LastError;
	__fastcall virtual TGIFImage(Classes::TComponent* aOwner);
	__fastcall virtual ~TGIFImage(void);
	virtual void __fastcall Repaint(void);
	__property Graphics::TBitmap* Bitmap = {read=GetBitmap};
	__property int BitsPerPixel = {read=FBitsPerPixel, nodefault};
	__property Classes::TMemoryStream* Data = {read=FData};
	__property bool Empty = {read=FEmpty, nodefault};
	__property int ImageWidth = {read=FImageWidth, nodefault};
	__property int ImageHeight = {read=FImageHeight, nodefault};
	__property bool IsAnimated = {read=FAnimated, nodefault};
	__property bool IsInterlaced = {read=FInterlaced, nodefault};
	__property bool IsTransparent = {read=FTransparent, nodefault};
	__property bool MouseOnTransparent = {read=FMouseOnTransparent, nodefault};
	__property int NumFrames = {read=FNumFrames, nodefault};
	__property int NumIterations = {read=FNumIterations, nodefault};
	virtual void __fastcall Assign(Classes::TPersistent* Source);
	void __fastcall Clear(void);
	TFrameInfo __fastcall GetFrameInfo(const int FrameIndex);
	bool __fastcall LoadFromFile(const AnsiString GIFFileName);
	bool __fastcall LoadFromStream(Classes::TStream* Stream);
	bool __fastcall LoadInfoFromFile(const AnsiString GIFFileName);
	bool __fastcall LoadInfoFromStream(Classes::TStream* Stream);
	bool __fastcall LoadFromResourceName(unsigned Instance, const AnsiString ResName);
	bool __fastcall LoadFromResourceID(unsigned Instance, const int ResID);
	
__published:
	__property Align ;
	__property Anchors ;
	__property Color ;
	__property Constraints ;
	__property Cursor ;
	__property DragCursor ;
	__property DragKind ;
	__property DragMode ;
	__property Height ;
	__property Hint ;
	__property Left ;
	__property ParentColor ;
	__property ParentShowHint ;
	__property PopupMenu ;
	__property ShowHint ;
	__property Top ;
	__property Width ;
	__property OnClick ;
	__property OnDblClick ;
	__property OnDragDrop ;
	__property OnDragOver ;
	__property OnEndDock ;
	__property OnEndDrag ;
	__property OnMouseDown ;
	__property OnMouseMove ;
	__property OnMouseUp ;
	__property OnStartDock ;
	__property OnStartDrag ;
	__property bool Animate = {read=FAnimate, write=SetAnimate, nodefault};
	__property bool AutoSize = {read=FAutoSize, write=SetAutoSize, nodefault};
	__property bool Center = {read=FCenter, write=SetCenter, nodefault};
	__property int CurrentFrame = {read=FCurrentFrame, write=SetFrame, stored=false, nodefault};
	__property bool DoubleBuffered = {read=FDoubleBuffered, write=FDoubleBuffered, default=1};
	__property bool FirstImageOnly = {read=FFirstImageOnly, write=FFirstImageOnly, nodefault};
	__property System::TObject* Graphic = {read=GetGraphic, write=SetGraphic};
	__property bool Loop = {read=FLoop, write=SetLoop, nodefault};
	__property bool Opaque = {read=FOpaque, write=SetOpaque, nodefault};
	__property int Speed = {read=FSpeed, write=SetSpeed, default=100};
	__property bool Stretch = {read=FStretch, write=SetStretch, nodefault};
	__property bool StretchRatio = {read=FStretchRatio, write=SetStretchRatio, nodefault};
	__property bool StretchBigOnly = {read=FStretchBigOnly, write=SetStretchBigOnly, nodefault};
	__property bool Tile = {read=FTile, write=SetTile, nodefault};
	__property bool Threaded = {read=FThreaded, write=SetThreaded, default=1};
	__property Classes::TThreadPriority ThreadPriority = {read=FThreadPriority, write=SetThreadPriority
		, default=3};
	__property bool Visible = {read=FVisible, write=SetVisible, default=1};
	__property TGIFVersion Version = {read=FVersion, nodefault};
	__property Classes::TNotifyEvent OnChanging = {read=FOnChanging, write=FOnChanging};
	__property Classes::TNotifyEvent OnChange = {read=FOnChange, write=FOnChange};
	__property TProgressEvent OnProgress = {read=FOnProgress, write=FOnProgress};
	__property Classes::TNotifyEvent OnWrapAnimation = {read=FOnWrapAnimation, write=FOnWrapAnimation};
		
};

#pragma pack(pop)

#pragma pack(push, 4)
class PASCALIMPLEMENTATION TGIFFrame : public System::TObject 
{
	typedef System::TObject inherited;
	
private:
	int Left;
	int Top;
	int Width;
	int Height;
	TGIFImage* Owner;
	int ColorDepth;
	int Delay;
	TDisposalType DisposalMethod;
	bool HasColorTable;
	bool IsInterlaced;
	bool IsTransparent;
	Byte TransparentColor;
	TRGBColor BackgroundColor;
	AnsiString Comment;
	TRGBColor *ColorTable;
	Graphics::TBitmap* FrameImage;
	Graphics::TBitmap* FrameImageMask;
	Graphics::TBitmap* PreviousImage;
	Graphics::TBitmap* PreviousImageMask;
	
protected:
	void __fastcall DecodeImage(Classes::TStream* DataStream, Classes::TStream* PixelStream, Classes::TStream* 
		MaskStream);
	bool __fastcall ReadImage(Classes::TStream* DataStream, TRGBColor * GlobalColorTable, void *GraphicControlBlock
		);
	
public:
	__fastcall TGIFFrame(TGIFImage* aOwner);
	__fastcall virtual ~TGIFFrame(void);
};

#pragma pack(pop)

//-- var, const, procedure ---------------------------------------------------
static const Word MAX_GIF_FRAME_COUNT = 0x100;
extern PACKAGE TGIFError LastWriteError;
extern PACKAGE TProgressEvent OnWriteProgress;
extern PACKAGE bool __fastcall SaveToFile(const AnsiString GIFFileName, TSaveInfo &FramesArray);
extern PACKAGE bool __fastcall SaveToFileSingle(const AnsiString GIFFileName, const Graphics::TBitmap* 
	aBitmap, const bool aInterlaced, const bool aTransparent, const Graphics::TColor aTransparentColor)
	;
extern PACKAGE bool __fastcall SaveToStream(Classes::TStream* Stream, TSaveInfo &FramesArray);
extern PACKAGE bool __fastcall SaveToStreamSingle(Classes::TStream* Stream, const Graphics::TBitmap* 
	aBitmap, const bool aInterlaced, const bool aTransparent, const Graphics::TColor aTransparentColor)
	;

}	/* namespace Gifimage */
#if !defined(NO_IMPLICIT_NAMESPACE_USE)
using namespace Gifimage;
#endif
#pragma option pop	// -w-

#pragma delphiheader end.
//-- end unit ----------------------------------------------------------------
#endif	// GIFImage
