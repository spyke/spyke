//----------------------------------------------------------------------------
// GIFImage.hpp - bcbdcc32 generated hdr (DO NOT EDIT) rev: 0
// From: GIFImage.pas
//----------------------------------------------------------------------------
#ifndef GIFImageHPP
#define GIFImageHPP
//----------------------------------------------------------------------------
#include <Menus.hpp>
#include <TimerEx.hpp>
#include <Controls.hpp>
#include <Classes.hpp>
#include <Graphics.hpp>
#include <SysUtils.hpp>
#include <Messages.hpp>
#include <Windows.hpp>
#include <System.hpp>
namespace Gifimage
{
//-- type declarations -------------------------------------------------------
enum TGIFVersion { gvGIF87a, gvGIF89a };

#pragma pack(push, 1)
struct TRGBColor
{
	Byte Red;
	Byte Green;
	Byte Blue;
} ;
#pragma pack(pop)

typedef TRGBColor TColorTable[256];

typedef TColorTable *PColorTable;

enum TDisposalType { dtUndefined, dtDoNothing, dtToBackground, dtToPrevious };

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
	System::AnsiString iiComment;
} ;

struct TSaveInfo
{
	int siNumFrames;
	TFrameInfo *siFrames[256];
	Word siNumLoops;
	bool siUseGlobalColorTable;
} ;

enum TGIFError { geNone, geCanceled, geInternal, geFileFormat, geTooManyColors, geIndexOutOfBounds, 
	geWindows, geFileNotFound, geResourceNotFound };

typedef void __fastcall (__closure *TProgressEvent)(System::TObject* Sender, const long BytesProcessed
	, const long BytesToProcess, int PercentageProcessed, bool &KeepOnProcessing);

class __declspec(delphiclass) TGIFFrame;
class __declspec(delphiclass) TGIFImage;
class __declspec(pascalimplementation) TGIFImage : public Controls::TGraphicControl
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
	TThreadPriority FThreadPriority;
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
	void __fastcall SetThreaded(bool p0);
	void __fastcall SetThreadPriority(Classes::TThreadPriority p0);
	void __fastcall SetAnimate(bool p0);
	void __fastcall SetAutoSize(bool p0);
	void __fastcall SetCenter(bool p0);
	void __fastcall SetFrame(int p0);
	void __fastcall SetLoop(bool p0);
	void __fastcall SetOpaque(const bool p0);
	void __fastcall SetSpeed(int p0);
	void __fastcall SetStretch(bool p0);
	void __fastcall SetStretchRatio(bool p0);
	void __fastcall SetTile(bool p0);
	HIDESBASE void __fastcall SetVisible(bool p0);
	System::TObject* __fastcall GetGraphic(void);
	void __fastcall SetGraphic(System::TObject* const p0);
	void __fastcall UpdateTimer(const bool p0, const int p1);
	Graphics::TBitmap* __fastcall GetBitmap(void);
	Windows::TRect __fastcall GetClipRect(void);
	void __fastcall UpdateOpacity(void);
	HIDESBASE MESSAGE void __fastcall WMPaint(Messages::TWMPaint &GIFImage_);
	HIDESBASE MESSAGE void __fastcall WMMouseMove(Messages::TWMMouse &GIFImage_);
	void __fastcall SetStretchBigOnly(const bool p0);
	
protected:
	virtual void __fastcall AssignTo(Classes::TPersistent* p0);
	virtual void __fastcall Loaded(void);
	virtual void __fastcall DefineProperties(Classes::TFiler* p0);
	virtual void __fastcall ReadData(Classes::TStream* p0);
	virtual void __fastcall WriteData(Classes::TStream* p0);
	virtual HPALETTE __fastcall GetPalette(void);
	virtual bool __fastcall ReadGIF(void);
	virtual void __fastcall Changing(void);
	virtual void __fastcall Change(void);
	virtual void __fastcall DoProgress(void);
	void __fastcall LocalClear(void);
	void __fastcall SetupBitmap(void);
	void __fastcall RemoveBitmap(int p0);
	void __fastcall PaintImage(HDC p0);
	virtual void __fastcall Paint(void);
	void __fastcall OnTrigger(System::TObject* p0);
	
public:
	TGIFError LastError;
	__fastcall virtual TGIFImage(Classes::TComponent* p0);
	__fastcall virtual ~TGIFImage(void);
	virtual void __fastcall Repaint(void);
	__property Graphics::TBitmap* Bitmap = {read=GetBitmap, nodefault};
	__property int BitsPerPixel = {read=FBitsPerPixel, nodefault};
	__property Classes::TMemoryStream* Data = {read=FData, nodefault};
	__property bool Empty = {read=FEmpty, nodefault};
	__property int ImageWidth = {read=FImageWidth, nodefault};
	__property int ImageHeight = {read=FImageHeight, nodefault};
	__property bool IsAnimated = {read=FAnimated, nodefault};
	__property bool IsInterlaced = {read=FInterlaced, nodefault};
	__property bool IsTransparent = {read=FTransparent, nodefault};
	__property bool MouseOnTransparent = {read=FMouseOnTransparent, nodefault};
	__property int NumFrames = {read=FNumFrames, nodefault};
	__property int NumIterations = {read=FNumIterations, nodefault};
	virtual void __fastcall Assign(Classes::TPersistent* p0);
	void __fastcall Clear(void);
	TFrameInfo __fastcall GetFrameInfo(const int p0);
	bool __fastcall LoadFromFile(const System::AnsiString p0);
	bool __fastcall LoadFromStream(Classes::TStream* p0);
	bool __fastcall LoadInfoFromFile(const System::AnsiString p0);
	bool __fastcall LoadInfoFromStream(Classes::TStream* p0);
	bool __fastcall LoadFromResourceName(int p0, const System::AnsiString p1);
	bool __fastcall LoadFromResourceID(int p0, const int p1);
	
__published:
	__property Align ;
	__property Color ;
	__property Cursor ;
	__property DragCursor ;
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
	__property OnEndDrag ;
	__property OnMouseDown ;
	__property OnMouseMove ;
	__property OnMouseUp ;
	__property OnStartDrag ;
	__property bool Animate = {read=FAnimate, write=SetAnimate, nodefault};
	__property bool AutoSize = {read=FAutoSize, write=SetAutoSize, nodefault};
	__property bool Center = {read=FCenter, write=SetCenter, nodefault};
	__property int CurrentFrame = {read=FCurrentFrame, write=SetFrame, stored=false, nodefault};
	__property bool DoubleBuffered = {read=FDoubleBuffered, write=FDoubleBuffered, default=1};
	__property bool FirstImageOnly = {read=FFirstImageOnly, write=FFirstImageOnly, nodefault};
	__property System::TObject* Graphic = {read=GetGraphic, write=SetGraphic, nodefault};
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

class __declspec(pascalimplementation) TGIFFrame : public System::TObject
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
	System::AnsiString Comment;
	TColorTable *ColorTable;
	Graphics::TBitmap* FrameImage;
	Graphics::TBitmap* FrameImageMask;
	Graphics::TBitmap* PreviousImage;
	Graphics::TBitmap* PreviousImageMask;
	
protected:
	void __fastcall DecodeImage(Classes::TStream* p0, Classes::TStream* p1, Classes::TStream* p2);
	bool __fastcall ReadImage(Classes::TStream* p0, TRGBColor * p1, void *p2);
	
public:
	__fastcall TGIFFrame(TGIFImage* p0);
	__fastcall virtual ~TGIFFrame(void);
};

//-- var, const, procedure ---------------------------------------------------
#define MAX_GIF_FRAME_COUNT (Word)(256)
extern TGIFError LastWriteError;
extern TProgressEvent OnWriteProgress;
extern bool __fastcall SaveToFile(const System::AnsiString p0, TSaveInfo &GIFImage_);
extern bool __fastcall SaveToFileSingle(const System::AnsiString p0, Graphics::TBitmap* const p1, const 
	bool p2, const bool p3, const Graphics::TColor p4);
extern bool __fastcall SaveToStream(Classes::TStream* p0, TSaveInfo &GIFImage_);
extern bool __fastcall SaveToStreamSingle(Classes::TStream* p0, Graphics::TBitmap* const p1, const bool 
	p2, const bool p3, const Graphics::TColor p4);

}	/* namespace Gifimage */
#if !defined(NO_IMPLICIT_NAMESPACE_USE)
using namespace Gifimage;
#endif
//-- end unit ----------------------------------------------------------------
#endif	// GIFImage
