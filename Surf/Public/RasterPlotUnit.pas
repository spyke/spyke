unit RasterPlotUnit;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, ExtCtrls;

type
  TRasterForm = class(TForm)
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);

  private
    rBMP : TBitmap;
    { Private declarations }
  public
    procedure UpdateRasterPlot;
    { Public declarations }
  end;

var
  RasterForm: TRasterForm;

implementation

uses SurfBawdMain;//why does this need to be here???

{$R *.DFM}

procedure TRasterForm.UpdateRasterPlot;
const PixelCountMax = 32768;

type pRGBTripleArray = ^TRGBTripleArray;
     TRGBTripleArray = ARRAY[0..PixelCountMax-1] OF TRGBTriple;

var Row:  pRGBTripleArray;
    j, index, offset, xpos, zoomfactor  :  integer;

begin
  rBMP := TBitmap.Create;
  try
    rBMP.PixelFormat := pf24bit;
    rBMP.Width:= RasterForm.Width;
    rBMP.Height:= RasterForm.Height;

    index:= SurfBawdForm.TrackBar.Position;
    offset := SurfBawdForm.m_SpikeTimeStamp[index];
    zoomfactor:= RasterForm.Width * 100; // ie. 10ms per pixel
    while (SurfBawdForm.m_SpikeTimeStamp[index] - offset) < zoomfactor do
    begin
      xpos := (SurfBawdForm.m_SpikeTimeStamp[index] - offset)div 100;
      for j := 0 to 4 do // raster lines five pixels high
      begin
        Row := rBMP.Scanline[((SurfBawdForm.m_SpikeClusterID[index]+ 1)* 5 + j)];
        with Row[xpos] do
        begin
          rgbtRed   := 0;
          rgbtGreen := 0;
          rgbtBlue  := 0;
        end;
      end;
     // Showmessage ('Timestamp = '+inttostr((SurfBawdForm.m_SpikeTimeStamp[index] - offset)div 100)
     //              + '; index'+inttostr(index));
      inc(index);
    end;

    // Display on screen
    Canvas.Draw(0,0,rBMP);
    finally
      rBMP.Free;
    end;
  end;

procedure TRasterForm.FormCreate(Sender: TObject);
begin
  {rBMP := TBitmap.Create;
  rBMP.PixelFormat := pf24bit;
  rBMP.Height:= RasterForm.Height;
  rBMP.Width:= RasterForm.Width;}
end;

procedure TRasterForm.FormDestroy(Sender: TObject);
begin
//  rBMP.Free; //?FreeImage?   FormHide event preferable?
end;

end.
