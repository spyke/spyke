object EPlateMainForm: TEPlateMainForm
  Left = 300
  Top = 179
  BorderStyle = bsDialog
  Caption = 'Polytrode impedance tester/electroplator'
  ClientHeight = 228
  ClientWidth = 404
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -6
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  OnClose = FormClose
  OnCreate = FormCreate
  PixelsPerInch = 96
  TextHeight = 13
  object GroupBox1: TGroupBox
    Left = 1
    Top = 7
    Width = 232
    Height = 203
    BiDiMode = bdLeftToRight
    Caption = 'Automatic'
    Color = clBtnFace
    DragKind = dkDock
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clHighlight
    Font.Height = -20
    Font.Name = 'MS Sans Serif'
    Font.Style = [fsBold]
    ParentBiDiMode = False
    ParentColor = False
    ParentFont = False
    TabOrder = 2
    object Label1: TLabel
      Left = 12
      Top = 31
      Width = 89
      Height = 13
      Caption = '1. Select electrode'
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlack
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentColor = False
      ParentFont = False
    end
    object Label10: TLabel
      Left = 9
      Top = 53
      Width = 145
      Height = 26
      Alignment = taRightJustify
      Caption = '2. For electroplate, set desired site impedance (kOhm)'
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlack
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentColor = False
      ParentFont = False
      WordWrap = True
    end
    object Label6: TLabel
      Left = 12
      Top = 84
      Width = 213
      Height = 55
      AutoSize = False
      BiDiMode = bdLeftToRight
      Caption = 
        '3. Click on either of the buttons below to start/stop the contro' +
        'ller cycling through all the electrode channels specified by the' +
        ' electrode configuration file.'
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlack
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentBiDiMode = False
      ParentColor = False
      ParentFont = False
      WordWrap = True
    end
    object PTestImp: TPanel
      Left = 5
      Top = 145
      Width = 129
      Height = 34
      BevelWidth = 2
      Caption = 'Test Impedances'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -13
      Font.Name = 'MS Sans Serif'
      Font.Style = [fsBold]
      ParentFont = False
      TabOrder = 4
      OnClick = PTestImpClick
    end
    object PPlate: TPanel
      Left = 136
      Top = 145
      Width = 90
      Height = 34
      BevelWidth = 2
      Caption = 'Electroplate'
      Enabled = False
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -13
      Font.Name = 'MS Sans Serif'
      Font.Style = [fsBold]
      ParentFont = False
      TabOrder = 0
      OnClick = PlateClick
    end
    object CElectrode: TComboBox
      Left = 105
      Top = 27
      Width = 120
      Height = 22
      Style = csOwnerDrawFixed
      DropDownCount = 5
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -11
      Font.Name = 'Arial'
      Font.Style = []
      ItemHeight = 16
      ParentFont = False
      TabOrder = 1
      OnChange = CElectrodeChange
    end
    object ProgressBar: TProgressBar
      Left = 3
      Top = 183
      Width = 226
      Height = 16
      Anchors = [akLeft, akBottom]
      Min = 0
      Max = 100
      Smooth = True
      Step = 1
      TabOrder = 2
    end
    object TargetImp: TSpinEdit
      Left = 160
      Top = 52
      Width = 65
      Height = 30
      EditorEnabled = False
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clHighlight
      Font.Height = -16
      Font.Name = 'MS Sans Serif'
      Font.Style = [fsBold]
      Increment = 10
      MaxValue = 1500
      MinValue = 100
      ParentFont = False
      TabOrder = 3
      Value = 350
      OnChange = ChanSelectChange
    end
    object DumpCSV: TCheckBox
      Left = 146
      Top = 123
      Width = 76
      Height = 17
      Caption = 'Save to file'
      Checked = True
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clHighlight
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
      State = cbChecked
      TabOrder = 5
    end
  end
  object StatusBar: TStatusBar
    Left = 0
    Top = 212
    Width = 404
    Height = 16
    Panels = <
      item
        Alignment = taCenter
        Width = 158
      end
      item
        Alignment = taCenter
        Width = 60
      end
      item
        Alignment = taCenter
        Width = 65
      end
      item
        Alignment = taCenter
        Width = 90
      end>
    SimplePanel = False
  end
  object Manual: TGroupBox
    Left = 237
    Top = 7
    Width = 164
    Height = 118
    Caption = 'Manual'
    DragKind = dkDock
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clHighlight
    Font.Height = -20
    Font.Name = 'MS Sans Serif'
    Font.Style = [fsBold]
    ParentFont = False
    TabOrder = 0
    object Label2: TLabel
      Left = 107
      Top = 24
      Width = 39
      Height = 26
      Alignment = taCenter
      Caption = 'MUX Channel'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clHighlight
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
      WordWrap = True
    end
    object Label9: TLabel
      Left = 64
      Top = 102
      Width = 3
      Height = 13
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clHighlight
      Font.Height = -8
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentColor = False
      ParentFont = False
      WordWrap = True
    end
    object ModeSelect: TRadioGroup
      Left = 5
      Top = 25
      Width = 92
      Height = 59
      Caption = 'Mode'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ItemIndex = 0
      Items.Strings = (
        'Off'
        'Impedance'
        'Electroplate')
      ParentFont = False
      TabOrder = 0
      OnClick = ModeSelectClick
    end
    object ChanSelect: TSpinEdit
      Left = 107
      Top = 50
      Width = 46
      Height = 34
      Enabled = False
      MaxValue = 56
      MinValue = 1
      TabOrder = 1
      Value = 1
      OnChange = ChanSelectChange
    end
    object ManualAcq: TButton
      Left = 6
      Top = 89
      Width = 35
      Height = 24
      Caption = 'Acq'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlack
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
      TabOrder = 2
      OnClick = ManualAcqClick
    end
    object Button1: TButton
      Left = 92
      Top = 89
      Width = 68
      Height = 24
      Caption = 'Save to file'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clHighlight
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
      TabOrder = 3
      OnClick = Button1Click
    end
    object Button2: TButton
      Left = 44
      Top = 89
      Width = 45
      Height = 24
      Caption = 'Spect'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clHighlight
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
      TabOrder = 4
    end
  end
  object DTDIO: TDTAcq32
    Left = 173
    Top = 0
    Width = 32
    Height = 32
    ControlData = {
      00000200560A00002B050000020000000344494F000000000000000000000000
      000000FFFFFFFFFFFFFFFFFFFF0000000000000000000000000000FFFFFFFFFF
      FFFFFFFFFF0000000000000000FFFFFFFFFFFF000000000000000000000000FF
      FF000000000000000000000000000000000000000000000000FFFFFFFF}
  end
  object DTAcq: TDTAcq32
    Left = 138
    Top = 0
    Width = 32
    Height = 32
    OnQueueDone = DTAcqQueueDone
    ControlData = {
      00000200560A00002B0500000200000003412F44000000000000000000000000
      000000FFFFFFFFFFFFFFFFFFFF0000000000000000000000000000FFFFFFFFFF
      FFFFFFFFFF0000000000000000FFFFFFFFFFFF000000000000000000000000FF
      FF000000000000000000000000000000000000000000000000FFFFFFFF}
  end
  object Setup: TGroupBox
    Left = 237
    Top = 125
    Width = 164
    Height = 85
    Caption = 'Setup'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clNavy
    Font.Height = -20
    Font.Name = 'MS Sans Serif'
    Font.Style = [fsBold]
    ParentFont = False
    TabOrder = 5
    object Label3: TLabel
      Left = 72
      Top = 66
      Width = 81
      Height = 13
      Caption = 'A/D Chans  Gain'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlack
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
    end
    object Label5: TLabel
      Left = 8
      Top = 68
      Width = 3
      Height = 13
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clGray
      Font.Height = -5
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
    end
    object CInputGain: TComboBox
      Left = 127
      Top = 44
      Width = 33
      Height = 21
      DropDownCount = 4
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clHighlight
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = [fsBold]
      ItemHeight = 13
      ParentFont = False
      TabOrder = 0
      Text = '0'
      OnChange = CAD_CGChange
    end
    object CADChan: TComboBox
      Left = 71
      Top = 44
      Width = 57
      Height = 21
      DropDownCount = 4
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clHighlight
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = [fsBold]
      ItemHeight = 13
      ParentFont = False
      TabOrder = 1
      Text = '0'
      OnChange = CAD_CGChange
    end
    object RadioGroup1: TRadioGroup
      Left = 4
      Top = 22
      Width = 63
      Height = 45
      Caption = 'Controller'
      Enabled = False
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlack
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ItemIndex = 1
      Items.Strings = (
        'LPT-1'
        'DTDIO')
      ParentFont = False
      TabOrder = 2
    end
    object CBoard: TComboBox
      Left = 71
      Top = 17
      Width = 89
      Height = 22
      Style = csOwnerDrawFixed
      DropDownCount = 5
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -11
      Font.Name = 'Arial'
      Font.Style = []
      ItemHeight = 16
      ParentFont = False
      TabOrder = 3
      OnChange = CBoardChange
    end
  end
  object DTDAC: TDTAcq32
    Left = 208
    Top = 0
    Width = 32
    Height = 32
    ControlData = {
      00000200560A00002B0500000200000003442F41000000000000000000000000
      000000FFFFFFFFFFFFFFFFFFFF0000000000000000000000000000FFFFFFFFFF
      FFFFFFFFFF0000000000000000FFFFFFFFFFFF000000000000000000000000FF
      FF000000000000000000000000000000000000000000000000FFFFFFFF}
  end
  object Timer: TTimer
    Enabled = False
    OnTimer = TimerTick
    Left = 106
    Top = 2
  end
end
