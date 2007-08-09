object Form1: TForm1
  Left = 226
  Top = 403
  Width = 870
  Height = 340
  Caption = 'Form1'
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  OnShow = FormShow
  PixelsPerInch = 96
  TextHeight = 13
  object Label1: TLabel
    Left = 224
    Top = 144
    Width = 193
    Height = 74
    Caption = 'Label1'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -53
    Font.Name = 'Arial Black'
    Font.Style = []
    ParentFont = False
  end
  object Label2: TLabel
    Left = 456
    Top = 144
    Width = 193
    Height = 74
    Caption = 'Label1'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -53
    Font.Name = 'Arial Black'
    Font.Style = []
    ParentFont = False
  end
  object DTDIN: TDTAcq32
    Left = 160
    Top = 32
    Width = 32
    Height = 32
    OnSSEventDone = DTDINSSEventDone
    ControlData = {
      00000200560A00002B0500000200000007445441637133320544543334300300
      030000000000000000000000FFFF01000000FFFFFDFF00000000000000000000
      00000800FEFF00000000FFFFFFFF0000000000000000FFFFFFFFFFFF00000000
      0000000000000000FFFF00000000000000000000000000000000000000000000
      0000FFFFFFFF}
  end
end
