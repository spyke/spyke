object Form1: TForm1
  Left = 388
  Top = 33
  Width = 444
  Height = 610
  Caption = 'Form1'
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
    Left = 56
    Top = 24
    Width = 34
    Height = 32
    Caption = '0:0'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -27
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
  end
  object Label2: TLabel
    Left = 16
    Top = 208
    Width = 201
    Height = 17
    AutoSize = False
    Caption = 'Filename: '
  end
  object Label4: TLabel
    Left = 24
    Top = 280
    Width = 393
    Height = 281
    AutoSize = False
    Caption = 'Parameter: '
    WordWrap = True
  end
  object Label3: TLabel
    Left = 16
    Top = 224
    Width = 201
    Height = 17
    AutoSize = False
    Caption = 'Display software version:  '
  end
  object Label5: TLabel
    Left = 16
    Top = 240
    Width = 201
    Height = 17
    AutoSize = False
    Caption = 'Bytes read: '
  end
  object CT: TDTAcq32
    Left = 8
    Top = 32
    Width = 32
    Height = 32
    OnSSEventDone = CTSSEventDone
    ControlData = {
      00000200560A00002B0500000200000003432F54054454333430000000000000
      0000000000000000FFFF01000000FFFFFEFF0000000000000000000000000800
      FEFF0100FEFFFFFFFFFF0000000000000000FFFFFFFFFFFF0000000000000000
      00000000FFFF000000000000000000000000000000000000000000000000FFFF
      FFFF}
  end
  object Button1: TButton
    Left = 40
    Top = 64
    Width = 75
    Height = 25
    Caption = 'Start timer'
    TabOrder = 1
    OnClick = Button1Click
  end
  object Button2: TButton
    Left = 40
    Top = 120
    Width = 75
    Height = 25
    Caption = 'Start DIN'
    TabOrder = 2
    OnClick = Button2Click
  end
  object DIN: TDTAcq32
    Left = 0
    Top = 104
    Width = 32
    Height = 32
    OnSSEventDone = DINSSEventDone
    ControlData = {
      00000200560A00002B050000020000000344494E054454333430000000000000
      0000000000000000FFFFFFFFFFFFFFFFFFFF0000000000000000000000000000
      FFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFF0000000000000000
      00000000FFFF000000000000000000000000000000000000000000000000FFFF
      FFFF}
  end
  object DIN2: TDTAcq32
    Left = 0
    Top = 136
    Width = 32
    Height = 32
    ControlData = {
      00000200560A00002B050000020000000444494E320000000000000000000000
      00000000FFFFFFFFFFFFFFFFFFFF0000000000000000000000000000FFFFFFFF
      FFFFFFFFFFFF0000000000000000FFFFFFFFFFFF000000000000000000000000
      FFFF000000000000000000000000000000000000000000000000FFFFFFFF}
  end
  object Button3: TButton
    Left = 40
    Top = 168
    Width = 75
    Height = 25
    Caption = 'Display param'
    TabOrder = 5
    OnClick = Button3Click
  end
end
