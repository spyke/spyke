object Form3: TForm3
  Left = 406
  Top = 103
  Width = 326
  Height = 107
  Caption = 'Test file CRC'
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  PixelsPerInch = 96
  TextHeight = 13
  object Label1: TLabel
    Left = 0
    Top = 32
    Width = 68
    Height = 13
    Caption = 'Running CRC:'
  end
  object Label2: TLabel
    Left = 72
    Top = 32
    Width = 105
    Height = 13
    AutoSize = False
    Color = clGray
    ParentColor = False
  end
  object Label3: TLabel
    Left = 73
    Top = 48
    Width = 104
    Height = 13
    AutoSize = False
    Color = clGray
    ParentColor = False
  end
  object Label4: TLabel
    Left = 24
    Top = 47
    Width = 44
    Height = 13
    Alignment = taRightJustify
    BiDiMode = bdLeftToRight
    Caption = 'File CRC:'
    ParentBiDiMode = False
  end
  object Button1: TButton
    Left = 192
    Top = 24
    Width = 75
    Height = 25
    Caption = 'Button1'
    TabOrder = 0
    OnClick = Button1Click
  end
end
