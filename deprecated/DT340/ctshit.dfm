object Form4: TForm4
  Left = 605
  Top = 207
  Width = 229
  Height = 235
  Caption = 'Time interval perception'
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  OnCreate = FormCreate
  PixelsPerInch = 96
  TextHeight = 13
  object Button1: TButton
    Left = 64
    Top = 8
    Width = 75
    Height = 25
    Caption = 'Press me!'
    TabOrder = 0
    OnClick = Button1Click
  end
  object ListBox1: TListBox
    Left = 32
    Top = 40
    Width = 161
    Height = 97
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ItemHeight = 24
    ParentFont = False
    TabOrder = 1
    Visible = False
  end
  object Button2: TButton
    Left = 64
    Top = 152
    Width = 97
    Height = 25
    Caption = 'View/hide results'
    TabOrder = 2
    OnClick = Button2Click
  end
end
