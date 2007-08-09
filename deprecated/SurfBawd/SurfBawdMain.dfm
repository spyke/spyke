object SurfBawdForm: TSurfBawdForm
  Left = 510
  Top = 341
  Width = 570
  Height = 344
  Caption = 'SurfBawd - Raw Waveform Analysis'
  Color = clBtnFace
  Constraints.MinHeight = 110
  Constraints.MinWidth = 570
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  Icon.Data = {
    0000010001002020100000000000E80200001600000028000000200000004000
    0000010004000000000080020000000000000000000000000000000000000000
    000000008000008000000080800080000000800080008080000080808000C0C0
    C0000000FF0000FF000000FFFF00FF000000FF00FF00FFFF0000FFFFFF008888
    8FFF888FFF88807700AAAAAAA770F78888888888F88780000AA00077AA38888F
    F8888888888880000A0007F7FAA8AAAFFFF87F8888F80000A00FFFFF77008FFA
    AFFF888888800000A0FF99FF0007107FAAFFFFF88880000A0899999371377777
    8AAFFFFF88F0000A899999777777777331AAFFFFF88F000A8999978888880000
    001AAFFFF8FFFFA800998878888800001300AAFFFFFFFFAF0998888888880103
    30000AAFFFFFFFAF0988888888880030007000AAFFFFFA9F0088888888880170
    77703F99AAFFAA90F088888888880730787788999AAAA070F088800088888837
    8887F99999900F00F0000FFF088887888887F999990000FFFF0FF0008888878F
    888709991700000FFFF0088888887788000088987300000FFF08888888887888
    0777088887300FF00FF00888888878877888888F880FF00080FFF08887877888
    88888F8700F003780FFFFF087877FF8F78888788300007780FFFFF087877FFFF
    78F87707003007780FFFFF088887FFFFFFF887010000003780FFF088F888FFFF
    F8780100001000313800088F8887FFFF80000000700000007F8788887F88FFFF
    000000000000000078788778888FFFFF0000000000000003778888888888FFF8
    7737030000000038778888FF8877FFFFF87777001070077778888888888FFFFF
    F8788703780007888778788FFFFF8FFFFF8787788737387888788888FFFF0000
    0000000000000000000000000000000000000000000000000000000000000000
    0000000000000000000000000000000000000000000000000000000000000000
    0000000000000000000000000000000000000000000000000000000000000000
    000000000000000000000000000000000000000000000000000000000000}
  KeyPreview = True
  OldCreateOrder = False
  OnClose = FormClose
  OnCreate = FormCreate
  OnKeyDown = FormKeyDown
  OnKeyUp = FormKeyUp
  PixelsPerInch = 96
  TextHeight = 13
  object FileStatsPanel: TPanel
    Left = 0
    Top = 0
    Width = 562
    Height = 62
    Align = alTop
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -27
    Font.Name = 'Arial'
    Font.Style = []
    ParentFont = False
    TabOrder = 0
    object Label1: TLabel
      Left = 6
      Top = 1
      Width = 57
      Height = 16
      Caption = 'Filename:'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -13
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
    end
    object Label2: TLabel
      Left = 6
      Top = 19
      Width = 36
      Height = 15
      Caption = 'Probe:'
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentColor = False
      ParentFont = False
    end
    object Label5: TLabel
      Left = 306
      Top = 26
      Width = 45
      Height = 15
      Caption = '# SVals:'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
    end
    object Label6: TLabel
      Left = 174
      Top = 1
      Width = 49
      Height = 16
      Caption = 'Filesize:'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -13
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
    end
    object FileNameLabel: TLabel
      Left = 64
      Top = 1
      Width = 107
      Height = 16
      AutoSize = False
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlue
      Font.Height = -13
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
      ParentShowHint = False
      ShowHint = True
    end
    object NProbes: TLabel
      Left = 41
      Top = 19
      Width = 3
      Height = 15
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlue
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentColor = False
      ParentFont = False
    end
    object NStim: TLabel
      Left = 242
      Top = 42
      Width = 3
      Height = 15
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlue
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentColor = False
      ParentFont = False
    end
    object NSpikes: TLabel
      Left = 242
      Top = 26
      Width = 3
      Height = 15
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlue
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentColor = False
      ParentFont = False
    end
    object Label7: TLabel
      Left = 150
      Top = 42
      Width = 90
      Height = 15
      Alignment = taRightJustify
      Caption = '# Stimulus runs:'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
    end
    object Label8: TLabel
      Left = 307
      Top = 42
      Width = 43
      Height = 15
      Caption = '# Msgs:'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
    end
    object NMsgs: TLabel
      Left = 354
      Top = 42
      Width = 3
      Height = 15
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlue
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentColor = False
      ParentFont = False
    end
    object NSVals: TLabel
      Left = 354
      Top = 26
      Width = 3
      Height = 15
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlue
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentColor = False
      ParentFont = False
    end
    object SampleRate: TLabel
      Left = 78
      Top = 19
      Width = 3
      Height = 15
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlue
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
    end
    object RecTimeLabel: TLabel
      Left = 405
      Top = 1
      Width = 4
      Height = 16
      Hint = 'min:sec.msec'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlue
      Font.Height = -13
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
      ParentShowHint = False
      ShowHint = True
    end
    object Label11: TLabel
      Left = 303
      Top = 1
      Width = 101
      Height = 16
      Caption = 'Recording period:'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -13
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
    end
    object DataFileSize: TLabel
      Left = 225
      Top = 1
      Width = 4
      Height = 16
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlue
      Font.Height = -13
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
    end
    object Label4: TLabel
      Left = 152
      Top = 26
      Width = 88
      Height = 15
      Alignment = taRightJustify
      Caption = '# Spike epochs:'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ParentFont = False
    end
    object CElectrode: TComboBox
      Left = 4
      Top = 34
      Width = 133
      Height = 22
      TabStop = False
      Style = csOwnerDrawFixed
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -12
      Font.Name = 'Arial'
      Font.Style = []
      ItemHeight = 16
      ParentFont = False
      TabOrder = 0
      OnChange = CElectrodeChange
    end
    object Button3: TButton
      Left = 480
      Top = 8
      Width = 75
      Height = 25
      Caption = 'Button3'
      TabOrder = 1
      OnClick = Button3Click
    end
  end
  object Waveforms: TPanel
    Left = 0
    Top = 62
    Width = 562
    Height = 235
    Align = alClient
    FullRepaint = False
    ParentShowHint = False
    ShowHint = False
    TabOrder = 1
    Visible = False
    object Label3: TLabel
      Left = 496
      Top = 184
      Width = 32
      Height = 13
      Caption = 'Label3'
    end
    object ToolBar1: TToolBar
      Left = 1
      Top = 17
      Width = 152
      Height = 24
      Align = alNone
      ButtonHeight = 20
      ButtonWidth = 24
      Caption = 'ToolBar1'
      Color = clBtnFace
      Images = SmlButtonImages
      Indent = 5
      ParentColor = False
      TabOrder = 1
      object tbStepBack: TToolButton
        Left = 5
        Top = 2
        AllowAllUp = True
        Caption = 'tbStepBack'
        Grouped = True
        ImageIndex = 0
        OnClick = tbStepBackClick
      end
      object tbReversePlay: TToolButton
        Left = 29
        Top = 2
        AllowAllUp = True
        Caption = 'tbReversePlay'
        Grouped = True
        ImageIndex = 1
        Style = tbsCheck
        OnClick = tbPlayClick
      end
      object tbPlay: TToolButton
        Left = 53
        Top = 2
        AllowAllUp = True
        Caption = 'tbPlay'
        Grouped = True
        ImageIndex = 3
        Style = tbsCheck
        OnClick = tbPlayClick
      end
      object tbStepFoward: TToolButton
        Left = 77
        Top = 2
        AllowAllUp = True
        Caption = 'tbStepFoward'
        Grouped = True
        ImageIndex = 4
        OnClick = tbStepFowardClick
      end
      object tbPause: TToolButton
        Left = 101
        Top = 2
        AllowAllUp = True
        Caption = 'tbPause'
        ImageIndex = 5
        Style = tbsCheck
        OnClick = tbPauseClick
      end
      object tbStop: TToolButton
        Left = 125
        Top = 2
        AllowAllUp = True
        Caption = 'tbStop'
        ImageIndex = 2
        OnClick = tbStopClick
      end
    end
    object ToolBar2: TToolBar
      Left = 154
      Top = 1
      Width = 203
      Height = 40
      Align = alNone
      AutoSize = True
      ButtonHeight = 38
      ButtonWidth = 39
      Caption = 'ToolBar2'
      Color = clBtnFace
      EdgeBorders = []
      Images = LrgButtonImages
      ParentColor = False
      TabOrder = 2
      object tbExportData: TToolButton
        Left = 0
        Top = 2
        Hint = 'Export thresholded spikes'
        Caption = 'tbExportData'
        Grouped = True
        ImageIndex = 0
        ParentShowHint = False
        ShowHint = True
        OnClick = tbExportDataClick
      end
      object tbChartWin: TToolButton
        Left = 39
        Top = 2
        Hint = 'Open chart window'
        Caption = 'tbChartWin'
        ImageIndex = 1
        ParentShowHint = False
        ShowHint = True
        OnClick = tbChartWinClick
      end
      object tbTemplateWin: TToolButton
        Left = 78
        Top = 2
        Hint = 'Generate spike templates'
        Caption = 'tbTemplateWin'
        ImageIndex = 2
        ParentShowHint = False
        ShowHint = True
        Style = tbsCheck
        OnClick = tbTemplateWinClick
      end
      object tbFindTemplates: TToolButton
        Left = 117
        Top = 2
        Hint = 'Search for spikes'
        Caption = 'tbFindTemplates'
        ImageIndex = 3
        ParentShowHint = False
        ShowHint = True
        OnClick = tbFindTemplatesClick
      end
      object tbExport2File: TToolButton
        Left = 156
        Top = 2
        Hint = 'Export sorted spiketimes'
        Caption = 'tbExport2File'
        Grouped = True
        ImageIndex = 4
        ParentShowHint = False
        ShowHint = True
        OnClick = tbExport2FileClick
      end
      object ToolButton3: TToolButton
        Left = 195
        Top = 2
        Width = 8
        Caption = 'ToolButton3'
        ImageIndex = 4
        Style = tbsDivider
      end
    end
    object GroupBox1: TGroupBox
      Left = 411
      Top = 1
      Width = 150
      Height = 142
      Caption = ' Export options '
      Color = clBtnFace
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clNavy
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentColor = False
      ParentFont = False
      TabOrder = 4
      object SpinEdit1: TSpinEdit
        Left = 5
        Top = 91
        Width = 53
        Height = 26
        Hint = 'Maximum number of spikes to export, according to threshold'
        AutoSize = False
        Enabled = False
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -13
        Font.Name = 'Arial'
        Font.Style = []
        Increment = 50
        MaxValue = 10000
        MinValue = 1
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 0
        Value = 1000
        OnChange = SpinEdit1Change
      end
      object cbuV: TCheckBox
        Left = 105
        Top = 76
        Width = 30
        Height = 17
        Hint = 'Check to export data in uV (lower resolution!)'
        Alignment = taLeftJustify
        Caption = 'uV'
        Color = clBtnFace
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentColor = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 1
      end
      object cbHeader: TCheckBox
        Left = 5
        Top = 78
        Width = 90
        Height = 12
        Hint = 'Check to export data in uV (lower resolution!)'
        Alignment = taLeftJustify
        BiDiMode = bdLeftToRight
        Caption = 'Include header'
        Checked = True
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentBiDiMode = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        State = cbChecked
        TabOrder = 2
      end
      object cbThreshFilter: TCheckBox
        Left = 59
        Top = 95
        Width = 82
        Height = 17
        Hint = 'Export up to  '#39'spinedit'#39' number of spikes according to threshold'
        Alignment = taLeftJustify
        BiDiMode = bdLeftToRight
        Caption = 'threshold filter'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentBiDiMode = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 3
        OnClick = cbThreshFilterClick
      end
      object cbExportTimes: TCheckBox
        Left = 5
        Top = 14
        Width = 130
        Height = 17
        Alignment = taLeftJustify
        BiDiMode = bdLeftToRight
        Caption = 'Spiketimes (threshold)'
        Checked = True
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentBiDiMode = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        State = cbChecked
        TabOrder = 4
        OnClick = cbThreshFilterClick
      end
      object cbExportSVALs: TCheckBox
        Left = 5
        Top = 46
        Width = 130
        Height = 13
        Alignment = taLeftJustify
        BiDiMode = bdLeftToRight
        Caption = 'Digital SVALs'
        Checked = True
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentBiDiMode = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        State = cbChecked
        TabOrder = 5
        OnClick = cbThreshFilterClick
      end
      object cbExportEEG: TCheckBox
        Left = 5
        Top = 60
        Width = 130
        Height = 17
        Alignment = taLeftJustify
        BiDiMode = bdLeftToRight
        Caption = 'EEG CRs'
        Enabled = False
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentBiDiMode = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 6
        OnClick = cbThreshFilterClick
      end
      object CheckBox1: TCheckBox
        Left = 5
        Top = 29
        Width = 130
        Height = 17
        Alignment = taLeftJustify
        BiDiMode = bdLeftToRight
        Caption = 'Spiketimes (template)'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentBiDiMode = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 7
        OnClick = cbThreshFilterClick
      end
      object Button2: TButton
        Left = 66
        Top = 114
        Width = 80
        Height = 17
        Caption = 'Export template'
        TabOrder = 8
        OnClick = Button2Click
      end
    end
    object TrackBar: TTrackBar
      Left = 2
      Top = 1
      Width = 152
      Height = 16
      DragCursor = crDefault
      Constraints.MaxHeight = 30
      Max = 100
      Orientation = trHorizontal
      ParentShowHint = False
      PageSize = 100
      Frequency = 10
      Position = 0
      SelEnd = 0
      SelStart = 0
      ShowHint = True
      TabOrder = 0
      TabStop = False
      ThumbLength = 12
      TickMarks = tmTopLeft
      TickStyle = tsNone
      OnChange = TrackBarChange
      OnKeyDown = TrackBarKeyDown
      OnKeyUp = TrackBarKeyUp
    end
    object MsgPanel: TPanel
      Left = 1
      Top = 193
      Width = 560
      Height = 41
      Align = alBottom
      BevelOuter = bvLowered
      Constraints.MaxHeight = 140
      Constraints.MinHeight = 18
      FullRepaint = False
      TabOrder = 5
      object MsgMemo: TMemo
        Left = 1
        Top = 1
        Width = 558
        Height = 39
        Cursor = crHandPoint
        Hint = 'Click to jump to this event, shift-click to select data range'
        Align = alClient
        BorderStyle = bsNone
        Color = clBlack
        Constraints.MinHeight = 18
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clLime
        Font.Height = 10
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        Lines.Strings = (
          'Surf message log')
        ParentFont = False
        ParentShowHint = False
        ReadOnly = True
        ScrollBars = ssVertical
        ShowHint = True
        TabOrder = 0
        OnKeyUp = TrackBarKeyUp
        OnMouseDown = MsgMemoMouseDown
        OnMouseUp = MsgMemoMouseUp
      end
    end
    object ToolBar3: TToolBar
      Left = 358
      Top = 1
      Width = 44
      Height = 70
      Align = alNone
      AutoSize = True
      BorderWidth = 1
      ButtonWidth = 20
      Caption = 'ToolBar3'
      Color = clBtnFace
      EdgeBorders = []
      Flat = True
      Images = DisplayToggleImages
      ParentColor = False
      TabOrder = 3
      object tbToglWform: TToolButton
        Left = 0
        Top = 0
        Hint = 'Toggle waveform display'
        Down = True
        ImageIndex = 0
        ParentShowHint = False
        ShowHint = True
        Style = tbsCheck
        OnClick = tbToglWformClick
      end
      object tbToglStatsWin: TToolButton
        Left = 20
        Top = 0
        Hint = 'Toggle waveform stats window'
        Caption = 'tbToglStatsWin'
        ImageIndex = 1
        ParentShowHint = False
        Wrap = True
        ShowHint = True
        Style = tbsCheck
        OnClick = tbToglStatsWinClick
      end
      object tbToglHistWin: TToolButton
        Left = 0
        Top = 22
        Caption = 'tbToglHistWin'
        ImageIndex = 4
        Style = tbsCheck
        OnClick = tbToglHistWinClick
      end
      object tbToglProbeGUI: TToolButton
        Left = 20
        Top = 22
        Hint = 'Toggle probe layout window'
        Caption = 'tbToglProbeGUI'
        ImageIndex = 3
        ParentShowHint = False
        Wrap = True
        ShowHint = True
        Style = tbsCheck
        OnClick = tbToglProbeGUIClick
      end
      object tbRasterPlot: TToolButton
        Left = 0
        Top = 44
        AutoSize = True
        Caption = 'tbRasterPlot'
        Enabled = False
        ImageIndex = 2
        Style = tbsCheck
        OnClick = tbRasterPlotClick
      end
      object tbToglISIHist: TToolButton
        Left = 20
        Top = 44
        Caption = 'tbToglISIHist'
        Style = tbsCheck
        OnClick = tbToglISIHistClick
      end
    end
    object GroupBox2: TGroupBox
      Left = 152
      Top = 41
      Width = 137
      Height = 104
      Caption = 'Event detection'
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clNavy
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
      TabOrder = 6
      object Label9: TLabel
        Left = 5
        Top = 12
        Width = 51
        Height = 10
        Caption = 'lockout radius'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clBlack
        Font.Height = -8
        Font.Name = 'Small Fonts'
        Font.Style = []
        ParentFont = False
      end
      object Button1: TButton
        Left = 61
        Top = 31
        Width = 72
        Height = 16
        Hint = 'Show lockout boundaries on probe'
        Caption = 'Show lockout'
        TabOrder = 0
        OnClick = Button1Click
      end
      object seLockRadius: TSpinEdit
        Left = 7
        Top = 22
        Width = 49
        Height = 26
        Hint = 'Radius of spatial lockout (µm)'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clBlack
        Font.Height = -13
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        MaxValue = 1000
        MinValue = 0
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 1
        Value = 50
        OnChange = seLockRadiusChange
      end
      object cbDetectRaw: TCheckBox
        Left = 58
        Top = 13
        Width = 75
        Height = 18
        Hint = 'Check to detect events on processed and/or upsampled data'
        Alignment = taLeftJustify
        Caption = 'On raw data'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clBlack
        Font.Height = -8
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 2
        OnClick = cbDetectRawClick
      end
      object rgThreshold: TRadioGroup
        Left = 8
        Top = 47
        Width = 125
        Height = 55
        Hint = 'Lockout entire probe for the specified duration'
        Caption = 'Thresholding'
        Color = clBtnFace
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clBlack
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ItemIndex = 0
        Items.Strings = (
          'Simple....'
          'Spatiotemporal')
        ParentColor = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 3
      end
      object seLockTime: TSpinEdit
        Left = 78
        Top = 62
        Width = 49
        Height = 19
        Hint = 'Lockout (µsec)'
        Font.Charset = ANSI_CHARSET
        Font.Color = clBlack
        Font.Height = -9
        Font.Name = 'Small Fonts'
        Font.Style = []
        Increment = 50
        MaxValue = 2000
        MinValue = 100
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 4
        Value = 1000
        OnChange = seLockTimeChange
      end
    end
    object GroupBox3: TGroupBox
      Left = 16
      Top = 41
      Width = 134
      Height = 104
      Caption = 'Signal processing '
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clNavy
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
      TabOrder = 7
      object Label10: TLabel
        Left = 56
        Top = 14
        Width = 50
        Height = 26
        Caption = 'Upsample factor'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentFont = False
        WordWrap = True
      end
      object lblUpsample: TLabel
        Left = 85
        Top = 27
        Width = 3
        Height = 13
        Hint = 'Effective sample rate'
        ParentShowHint = False
        ShowHint = True
      end
      object seFactor: TSpinEdit
        Left = 7
        Top = 14
        Width = 45
        Height = 26
        AutoSize = False
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -13
        Font.Name = 'Arial'
        Font.Style = []
        MaxValue = 100
        MinValue = 1
        ParentFont = False
        ParentShowHint = False
        ShowHint = False
        TabOrder = 0
        Value = 4
        OnChange = seFactorChange
      end
      object cbSHcorrect: TCheckBox
        Left = 29
        Top = 46
        Width = 91
        Height = 12
        Hint = 'Check to correct for ADC S:H delays'
        Alignment = taLeftJustify
        Caption = 'S&&H corrected'
        Checked = True
        Color = clBtnFace
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentColor = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        State = cbChecked
        TabOrder = 1
        OnClick = cbSHcorrectClick
      end
      object cbPCAClean: TCheckBox
        Left = 37
        Top = 73
        Width = 83
        Height = 12
        Hint = 'Not yet implemented'
        Alignment = taLeftJustify
        Caption = 'PCA cleaner'
        Color = clBtnFace
        Enabled = False
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentColor = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 2
      end
      object cbAlignData: TCheckBox
        Left = 25
        Top = 59
        Width = 95
        Height = 12
        Hint = 'Corrects board asynchrony for cat 9 data'
        Alignment = taLeftJustify
        Caption = 'Cat 9 data align'
        Color = clBtnFace
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentColor = False
        ParentFont = False
        ParentShowHint = False
        ShowHint = True
        TabOrder = 3
      end
      object Button4: TButton
        Left = 48
        Top = -8
        Width = 75
        Height = 25
        Caption = 'n per tetrode'
        TabOrder = 4
        OnClick = Button4Click
      end
      object Button5: TButton
        Left = -48
        Top = 48
        Width = 75
        Height = 25
        Caption = 'show 2d posn'
        TabOrder = 5
        OnClick = Button5Click
      end
      object Button6: TButton
        Left = -48
        Top = 72
        Width = 75
        Height = 25
        Caption = 'Export tet wav'
        TabOrder = 6
        OnClick = Button6Click
      end
      object Button7: TButton
        Left = -24
        Top = -8
        Width = 75
        Height = 25
        Caption = 'Char2Clipbrd'
        TabOrder = 7
        OnClick = Button7Click
      end
      object Button8: TButton
        Left = 88
        Top = 8
        Width = 51
        Height = 25
        Caption = 'ch per n'
        TabOrder = 8
        OnClick = Button8Click
      end
      object Button9: TButton
        Left = 88
        Top = 32
        Width = 57
        Height = 25
        Caption = 'showFields'
        TabOrder = 9
        OnClick = Button9Click
      end
      object cbDecimate: TCheckBox
        Left = 48
        Top = 85
        Width = 97
        Height = 17
        Caption = '12.5kHz'
        TabOrder = 10
      end
    end
    object GroupBox4: TGroupBox
      Left = 296
      Top = 69
      Width = 113
      Height = 74
      Caption = 'Spike counter '
      TabOrder = 8
      object TLabel
        Left = 8
        Top = 16
        Width = 55
        Height = 24
        Caption = 'Count:'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -19
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentFont = False
      end
      object lbSpikeCount: TLabel
        Left = 65
        Top = 16
        Width = 10
        Height = 24
        Caption = '0'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -19
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentFont = False
      end
      object btReset: TButton
        Left = 8
        Top = 41
        Width = 97
        Height = 25
        Caption = 'Reset'
        TabOrder = 0
        OnClick = btResetClick
      end
    end
    object ClusterLog: TMemo
      Left = 16
      Top = 152
      Width = 473
      Height = 105
      Lines.Strings = (
        'ClusterLog')
      ReadOnly = True
      TabOrder = 9
    end
    object SpinEdit2: TSpinEdit
      Left = 496
      Top = 152
      Width = 65
      Height = 22
      MaxValue = 0
      MinValue = 0
      TabOrder = 10
      Value = 0
      OnChange = SpinEdit2Change
    end
  end
  object StatusBar: TStatusBar
    Left = 0
    Top = 297
    Width = 562
    Height = 20
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -12
    Font.Name = 'Tahoma'
    Font.Style = []
    Panels = <
      item
        Text = 'Time: '
        Width = 120
      end
      item
        Width = 150
      end
      item
        Text = 'No range selected'
        Width = 50
      end>
    SimplePanel = False
    UseSystemFont = False
    Visible = False
    OnDblClick = StatusBarDblClick
  end
  object SurfFile: TSurfFileAccess
    OnNewFile = SurfFileNewFile
    Left = 462
    Top = 30
  end
  object SmlButtonImages: TImageList
    Height = 14
    Width = 17
    Left = 407
    Top = 30
    Bitmap = {
      494C010106000900040011000E00FFFFFFFFFF10FFFFFFFFFFFFFFFF424D3600
      0000000000003600000028000000440000002A0000000100200000000000A02C
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000090908F0090908F0090908F009090
      8F0090908F0090908F0090908F0090908F0090908F0090908F0090908F009090
      8F0090908F0090908F0090908F0090908F0090908F0000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000A0A09F00AFAFAF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0
      AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00AFAF
      AF00AFAFAF00AFAFAF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0
      AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0
      AF00AFAFAF00AFA0A00000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000AFAFAF00BFBFB000C0C0BF00F0F0
      E000F0F0E000C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00F0F0E000F0F0
      E000F0F0E000C0C0BF00BFBFB000B0B0AF00AFAFAF00AFAFAF00B0B0AF00C0C0
      BF00EFEFE000EFEFE000FFFFEF00EFEFE000C0C0BF00C0C0BF00C0C0BF00EFEF
      E000FFFFEF00EFEFE000FFFFEF00BFBFB000BFBFB000AFAFAF00000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000AFAFAF00BFBFBF00C0C0BF001F101F00AFAFAF00F0F0E000C0C0C000C0C0
      BF00C0C0C000C0C0BF001F101F00A0A09F00F0F0E000C0C0BF00C0C0BF00BFBF
      B000B0B0AF00B0B0AF00BFBFB000C0C0BF001F101F00A09F9F00A09F9F00FFFF
      EF00C0C0BF00CFCFC000CFCFC0001F101F00A09F9F00A09F9F00FFFFEF00C0C0
      BF00BFBFB000AFAFAF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000B0B0AF00C0C0B000C0C0BF001F10
      1F00A0A09F00A0A09F00F0F0E000CFCFC000CFCFC000CFCFC0001F101F00A0A0
      9F00F0F0E000CFCFC000C0C0BF00BFBFB000B0B0AF00B0B0AF00BFBFB000C0C0
      BF001F101F00A09F9F00AFA0A000FFFFEF00CFCFC000CFCFC000CFCFC000100F
      1000A09F9F00AFA0A000FFFFEF00CFCFC000C0C0BF00B0B0AF00000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000B0B0AF00BFBFBF00CFCFC0001F101F00A0A09F00A0A09F00A0A09F00F0F0
      E000CFCFC000CFCFC0001F101F00A0A09F00F0F0E000CFCFC000C0C0BF00BFBF
      B000B0B0AF00B0B0AF00BFBFB000CFCFC000100F1000A09F9F00AFA0A000FFFF
      EF00CFCFC000CFCFC000CFCFC000100F1000A09F9F00AFA0A000FFFFEF00CFCF
      C000C0C0BF00B0B0AF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000B0B0AF00C0C0BF00C0C0BF001F10
      1F00A0A09F00A0A09F00A0A09F00A0A09F00F0F0E000CFCFC0001F101F00A0A0
      9F00F0F0E000CFCFC000C0C0BF00BFBFBF00B0B0AF00B0B0AF00BFBFB000C0C0
      BF001F101F00A09F9F00AFA0A000FFFFEF00CFCFC000CFCFC000CFCFC0001F10
      1F00A09F9F00AFA0A000FFFFEF00C0C0BF00C0C0BF00B0B0AF00000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000B0B0AF00C0C0BF00C0C0C0001F101F00A0A09F00A0A09F00A0A09F009090
      8F001F101F00BFB0B0001F101F00A0A09F00F0F0E000CFCFC000C0C0BF00BFBF
      B000B0B0AF00B0B0AF00BFBFB000C0C0BF001F101F00A09F9F00AFA0A000FFFF
      EF00CFCFC000CFCFC000CFCFC000100F1000A09F9F00AFA0A000FFFFEF00CFCF
      C000C0C0BF00B0B0AF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000B0B0AF00C0C0BF00CFCFBF001F10
      1F00A0A09F00A0A09F0090908F001F101F00CFCFC000CFCFC0001F101F00A0A0
      9F00F0F0E000CFCFC000C0C0BF00BFBFB000B0B0B000B0B0AF00C0C0BF00C0C0
      BF00100F1000A09F9F00AFA0A000FFFFEF00CFCFC000CFCFC000CFCFC0001F10
      1F00A09F9F00AFA0A000FFFFEF00CFCFC000C0C0BF00B0B0AF00000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000B0B0AF00C0C0BF00C0C0C0001F101F00A0A09F0090908F001F101F00CFCF
      C000CFCFC000CFCFC0001F101F00A0A09F00F0F0E000CFCFC000C0C0BF00BFBF
      BF00B0B0AF00B0B0B000BFBFB000CFCFC000100F1000A09F9F00AFA0A000FFFF
      EF00CFCFC000CFCFC000CFCFC000100F1000A09F9F00AFA0A000FFFFEF00CFCF
      C000C0C0BF00B0B0AF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000B0B0AF00C0C0BF00C0C0BF001F10
      1F00A0A09F001F101F00CFCFC000CFCFC000CFCFC000CFCFC0001F101F00A0A0
      9F00F0F0E000CFCFC000C0C0BF00C0C0B000BFBFB000BFBFB000C0C0BF00C0C0
      BF001F101F00FFFFFF00A09F9F00FFFFEF00CFCFC000CFCFC000CFCFC0001F10
      1F0000000000A09F9F00FFFFEF00C0C0BF00C0C0BF00B0B0AF00000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000B0B0AF00C0C0B000C0C0C0001F101F001F101F00C0C0BF00C0C0C000C0C0
      BF00C0C0C000CFCFC0001F101F001F101F001F101F00C0C0BF00C0C0BF00BFBF
      BF00BFBFB000BFBFB000C0C0BF00C0C0BF00100F1000100F10001F101F001F10
      1F00C0C0BF00CFCFC000C0C0BF00100F1000100F10001F101F001F101F00CFCF
      C000C0C0BF00B0B0AF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000AFAFAF00BFBFB000BFBFB000C0C0
      BF00BFBFB000C0C0BF00C0C0B000C0C0BF00BFBFB000C0C0BF00C0C0BF00BFBF
      BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0
      BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00BFBFB000C0C0BF00C0C0
      BF00C0C0BF00BFBFB000C0C0BF00BFBFB000BFBFB000AFAFAF00000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000AFAFAF00AFAFAF00B0B0AF00B0B0AF00BFB0B000B0B0AF00B0B0B000B0B0
      B000BFB0B000B0B0AF00B0B0B000BFBFB000BFBFB000BFBFB000BFBFBF00C0C0
      BF00CFCFC000CFCFC000C0C0BF00C0C0BF00BFBFB000BFBFB000BFBFB000B0B0
      B000B0B0AF00BFB0B000B0B0B000B0B0B000B0B0AF00BFB0B000B0B0AF00B0B0
      AF00AFAFAF00AFAFAF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000090908F0090908F0090908F009090
      8F0090908F0090908F0090908F0090908F0090908F0090908F0090908F009090
      8F0090908F0090908F0090908F0090908F0090908F00908F8F00908F8F00908F
      8F00908F8F00908F8F00908F8F00908F8F00908F8F00908F8F00908F8F00908F
      8F00908F8F00908F8F00908F8F00908F8F009F908F00908F8F00000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000000000000000908F
      8F009F908F00908F8F00908F8F00908F8F00908F8F00908F8F00908F8F00908F
      8F00908F8F00908F8F00908F8F00908F8F00908F8F00908F8F00908F8F00908F
      8F00AFAFAF00AFAFAF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0
      AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00AFAF
      AF00A0A09F00A09F9F00AFAFA000B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0
      AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0
      AF00AFAFA000AFAFA000AFAFAF00AFAFAF00B0B0AF00B0B0AF00B0B0AF00B0B0
      AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0
      AF00B0B0AF00AFAFAF00A0A09F00AFAFA000AFAFA000B0B0AF00B0B0AF00B0B0
      AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0AF00B0B0
      AF00B0B0AF00B0B0AF00AFAFA000A09F9F00AFAFAF00B0B0AF00BFBFB000C0C0
      BF00F0F0E000F0F0E000F0F0E000C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0
      BF00F0F0E000F0F0E000C0C0BF00BFBFB000AFAFAF00AFAFA000BFB0B000BFB0
      B000C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0
      BF00EFEFDF00EFEFDF00C0C0BF00BFB0B000B0B0AF00AFAFA000AFAFAF00B0B0
      AF00BFBFBF00C0C0BF00E0E0D000F0F0E000F0F0E000F0F0E000F0F0E000F0F0
      E000F0F0E000F0F0E000F0F0E000F0F0E000C0C0BF00BFBFBF00AFAFAF00AFAF
      A000B0B0AF00BFB0B000C0C0BF00EFEFDF00EFEFDF00C0C0BF00C0C0BF00C0C0
      BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00BFB0B000BFB0B000AFAF
      A000B0B0AF00BFBFB000C0C0BF00C0C0BF001F101F00A0A09F00F0F0E000C0C0
      BF00C0C0C000C0C0BF00C0C0C000F0F0E000AFAFAF00F0F0E000C0C0BF00BFBF
      BF00AFAFAF00AFAFA000BFBFBF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00CFCF
      C000C0C0BF00CFCFC000EFEFDF00F0F0E000A09F9F00F0F0E000C0C0BF00C0C0
      BF00BFB0B000B0B0AF00B0B0AF00BFBFBF00C0C0BF00C0C0BF001F101F00AFAF
      AF00A09F9F00A0A09F00A0A09F00A0A09F00A09F9F00AFAFAF00A0A09F00FFFF
      EF00C0C0BF00BFBFBF00AFAFAF00B0B0AF00BFB0B000C0C0BF00C0C0BF000F0F
      0F00A09F9F00F0F0E000EFEFDF00CFCFC000C0C0BF00CFCFC000C0C0BF00C0C0
      BF00C0C0BF00C0C0BF00BFBFBF00AFAFA000B0B0AF00BFBFB000C0C0BF00CFCF
      C0001F101F00A0A09F00F0F0E000CFCFC000CFCFC000CFCFC000F0F0E000A0A0
      9F00A0A09F00F0F0E000C0C0BF00C0C0B000B0B0AF00B0B0AF00C0C0B000C0C0
      C000CFCFC000CFCFC000CFCFC000CFCFC000F0F0E000F0F0E000A09F9F00A09F
      9F009F908F00F0F0E000CFCFC000C0C0BF00BFB0B000B0B0AF00B0B0AF00BFBF
      BF00C0C0BF00CFCFC000100F1000A09F9F00A09F9F00A09F9F00A09F9F00A09F
      9F00A09F9F00A09F9F00A0A09F00FFFFEF00C0C0BF00C0C0B000B0B0AF00B0B0
      AF00BFB0B000C0C0BF00CFCFC0000F0F0F009F908F00A09F9F00A09F9F00F0F0
      E000F0F0E000CFCFC000CFCFC000CFCFC000CFCFC000C0C0C000C0C0B000B0B0
      AF00B0B0AF00BFBFB000C0C0BF00CFCFC0001F101F00A0A09F00F0F0E000CFCF
      C000CFCFC000F0F0E000A0A09F00A0A09F00A0A09F00F0F0E000CFCFC000BFBF
      BF00B0B0AF00B0B0AF00BFBFBF00CFCFBF00CFCFC000CFCFC000F0F0E000F0F0
      E000A09F9F00A09F9F009F908F009F908F009F908F00F0F0E000CFCFC000C0C0
      BF00BFB0B000B0B0AF00B0B0AF00BFBFBF00C0C0BF00CFCFC000100F1000A09F
      9F00A09F9F00A09F9F00A09F9F00A09F9F00A09F9F00A09F9F00AFAFAF00FFFF
      EF00CFCFC000BFBFBF00B0B0AF00B0B0AF00BFB0B000C0C0BF00CFCFC0000F0F
      0F009F908F009F908F009F908F00A09F9F00A09F9F00F0F0E000F0F0E000CFCF
      C000CFCFC000CFCFBF00BFBFBF00B0B0AF00B0B0AF00BFBFBF00C0C0BF00CFCF
      C0001F101F00A0A09F00F0F0E000CFCFC000F0F0E000A0A09F00A0A09F00A0A0
      9F00A0A09F00F0F0E000C0C0BF00C0C0BF00B0B0AF00B0B0AF00C0C0BF00CFCF
      C000EFEFDF00F0F0E000A09F9F00A09F9F009F908F009F908F009F908F009F90
      8F009F908F00F0F0E000CFCFC000C0C0BF00BFB0B000B0B0AF00B0B0AF00BFBF
      BF00C0C0BF00CFCFC0001F101F00A09F9F00A09F9F00A09F9F00A09F9F00A09F
      9F00A09F9F00A09F9F00AFAFAF00FFFFEF00C0C0BF00C0C0BF00B0B0AF00B0B0
      AF00BFB0B000C0C0BF00CFCFC0000F0F0F009F908F009F908F009F908F009F90
      8F009F908F00A09F9F00A09F9F00F0F0E000EFEFDF00CFCFC000C0C0BF00B0B0
      AF00B0B0AF00BFBFB000C0C0BF00CFCFC0001F101F00A0A09F00F0F0E000BFB0
      B0001F101F0090908F00A0A09F00A0A09F00A0A09F00F0F0E000C0C0C000C0C0
      BF00B0B0AF00B0B0AF00BFBFBF00CFCFC0000F0F0F000F0F0F00908F8F009F90
      8F009F908F009F908F009F908F009F908F009F908F00F0F0E000CFCFC000C0C0
      BF00BFB0B000B0B0AF00B0B0AF00BFBFBF00C0C0BF00CFCFC000100F1000A09F
      9F00A09F9F00A09F9F00A09F9F00A09F9F00A09F9F00A09F9F00AFAFAF00FFFF
      EF00CFCFC000C0C0BF00B0B0AF00B0B0AF00BFB0B000C0C0BF00CFCFC0000F0F
      0F009F908F009F908F009F908F009F908F009F908F009F908F00908F8F000F0F
      0F000F0F0F00CFCFC000BFBFBF00B0B0AF00B0B0B000BFBFB000C0C0BF00CFCF
      C0001F101F00A0A09F00F0F0E000CFCFC000CFCFC0001F101F0090908F00A0A0
      9F00A0A09F00F0F0E000CFCFBF00C0C0BF00B0B0AF00B0B0AF00C0C0BF00CFCF
      BF00CFCFC000CFCFC0000F0F0F000F0F0F00908F8F009F908F009F908F009F90
      8F009F908F00F0F0E000CFCFC000C0C0BF00BFB0B000BFB0B000BFB0B000BFBF
      BF00C0C0BF00CFCFC000100F1000A09F9F00A09F9F00A09F9F00A09F9F00A09F
      9F00A09F9F00A09F9F00AFAFAF00FFFFEF00CFCFC000C0C0BF00B0B0AF00BFB0
      B000BFB0B000C0C0BF00CFCFC0000F0F0F009F908F009F908F009F908F009F90
      8F00908F8F000F0F0F000F0F0F00CFCFC000CFCFC000CFCFBF00C0C0BF00B0B0
      AF00B0B0AF00BFBFBF00C0C0BF00CFCFC0001F101F00A0A09F00F0F0E000CFCF
      C000CFCFC000CFCFC0001F101F0090908F00A0A09F00F0F0E000C0C0C000C0C0
      BF00B0B0AF00B0B0AF00C0C0BF00C0C0C000CFCFC000CFCFC000CFCFC000CFCF
      C0000F0F0F000F0F0F00908F8F009F908F009F908F00F0F0E000CFCFC000C0C0
      BF00BFBFBF00B0B0AF00B0B0AF00BFBFBF00C0C0BF00CFCFC0001F101F00A09F
      9F00A09F9F00A09F9F00A09F9F00A09F9F00A09F9F00A09F9F00AFAFAF00FFFF
      EF00CFCFC000C0C0BF00B0B0AF00B0B0AF00BFBFBF00C0C0BF00CFCFC0000F0F
      0F009F908F009F908F00908F8F000F0F0F000F0F0F00CFCFC000CFCFC000CFCF
      C000CFCFC000C0C0C000C0C0BF00B0B0AF00BFBFB000C0C0B000C0C0BF00CFCF
      C0001F101F00A0A09F00F0F0E000CFCFC000CFCFC000CFCFC000CFCFC0001F10
      1F00A0A09F00F0F0E000C0C0BF00C0C0BF00B0B0AF00B0B0AF00C0C0BF00C0C0
      BF00CFCFC000CFCFC000CFCFC000CFCFC000CFCFC000CFCFC0000F0F0F000F0F
      0F00908F8F00F0F0E000CFCFC000C0C0BF00C0C0B000BFB0B000BFBFBF00C0C0
      B000C0C0BF00CFCFC000100F1000A09F9F00A09F9F00A09F9F00A09F9F00A09F
      9F00A09F9F00A09F9F00A09F9F00F0F0E000C0C0BF00C0C0BF00B0B0AF00BFB0
      B000C0C0B000C0C0BF00CFCFC0000F0F0F00908F8F000F0F0F000F0F0F00CFCF
      C000CFCFC000CFCFC000CFCFC000CFCFC000CFCFC000C0C0BF00C0C0BF00B0B0
      AF00BFBFB000BFBFBF00C0C0BF00C0C0BF001F101F001F101F001F101F00CFCF
      C000C0C0C000C0C0BF00C0C0C000C0C0BF001F101F001F101F00C0C0C000C0C0
      B000B0B0AF00B0B0AF00C0C0B000C0C0BF00C0C0C000C0C0BF00C0C0C000CFCF
      BF00C0C0C000CFCFBF00C0C0C000C0C0BF000F0F0F000F0F0F00C0C0BF00C0C0
      BF00BFBFBF00BFB0B000BFBFBF00BFBFBF00C0C0BF00C0C0BF00100F10001F10
      1F00100F1000100F10001F101F00100F1000100F1000100F10001F101F001F10
      1F00CFCFC000C0C0B000B0B0AF00BFB0B000BFBFBF00C0C0BF00C0C0BF000F0F
      0F000F0F0F00C0C0BF00C0C0C000CFCFBF00C0C0C000CFCFBF00C0C0C000C0C0
      BF00C0C0C000C0C0BF00C0C0B000B0B0AF00C0C0BF00C0C0BF00C0C0BF00C0C0
      BF00C0C0BF00BFBFBF00C0C0BF00C0C0BF00BFBFB000C0C0BF00C0C0B000C0C0
      BF00BFBFB000C0C0BF00BFBFB000BFBFB000AFAFAF00B0B0AF00BFB0B000BFB0
      B000C0C0BF00BFB0B000C0C0B000BFBFBF00C0C0B000BFBFBF00C0C0B000C0C0
      BF00BFBFBF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00C0C0
      BF00C0C0BF00C0C0BF00C0C0BF00BFBFBF00C0C0BF00C0C0BF00BFBFBF00C0C0
      BF00C0C0B000C0C0BF00BFBFBF00C0C0BF00BFBFBF00BFBFBF00AFAFAF00C0C0
      BF00C0C0BF00C0C0BF00C0C0BF00C0C0BF00BFBFBF00C0C0BF00C0C0B000BFBF
      BF00C0C0B000BFBFBF00C0C0B000BFB0B000C0C0BF00BFB0B000BFB0B000B0B0
      AF00CFCFC000C0C0BF00BFBFBF00BFBFB000BFBFB000BFBFB000B0B0B000B0B0
      AF00BFB0B000B0B0B000B0B0B000B0B0AF00BFB0B000B0B0AF00B0B0AF00AFAF
      AF00AFAFAF00AFAFA000B0B0AF00B0B0AF00B0B0AF00BFB0B000BFB0B000BFB0
      B000BFB0B000BFB0B000BFB0B000B0B0AF00BFB0B000BFB0B000BFB0B000BFBF
      BF00C0C0BF00CFCFC000CFCFC000C0C0BF00BFBFBF00BFBFBF00BFBFBF00BFBF
      BF00BFB0B000B0B0AF00BFB0B000BFB0B000BFB0B000B0B0AF00BFB0B000B0B0
      AF00B0B0AF00AFAFAF00AFAFAF00CFCFC000C0C0BF00BFBFBF00BFB0B000BFB0
      B000BFB0B000B0B0AF00BFB0B000BFB0B000BFB0B000BFB0B000BFB0B000BFB0
      B000B0B0AF00B0B0AF00B0B0AF00AFAFA000424D3E000000000000003E000000
      28000000440000002A0000000100010000000000F80100000000000000000000
      000000000000000000000000FFFFFF0000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000000000007FFFC0000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000008000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000003FFFE000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000000000000000}
  end
  object LrgButtonImages: TImageList
    Height = 32
    Width = 32
    Left = 423
    Top = 30
    Bitmap = {
      494C010105000900040020002000FFFFFFFFFF10FFFFFFFFFFFFFFFF424D3600
      00000000000036000000280000008000000060000000010020000000000000C0
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FF00FF0000000000FF00
      FF0000000000FF00FF0000000000FF00FF00FF00FF0000000000FF00FF000000
      0000FF00FF00FF00FF00FF00FF0000000000FF00FF0000000000544E3F00544E
      3F00544E3F00544E3F00544E3F00544E3F00544E3F00544E3F00544E3F00544E
      3F00544E3F00544E3F00544E3F00000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FF00FF0000000000FF00
      FF0000000000FF00FF0000000000FF00FF00FF00FF0000000000FF00FF000000
      0000FF00FF00FF00FF00FF00FF0000000000FF00FF0086806D0086806D008680
      6D0086806D0086806D00544E3F00544E3F00544E3F00544E3F00544E3F008680
      6D0086806D0086806D0086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FF00FF0000000000FF00
      FF0000000000FF00FF0000000000FF00FF00FF00FF0000000000FF00FF000000
      0000FF00FF00FF00FF00FF00FF0000000000FF00FF0086806D0000000000BEB5
      9800BEB59800BEB5980086806D0086806D0086806D0086806D00544E3F00D8CE
      B90000000000A99E750086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FF00FF0000000000FF00
      FF0000000000FF00FF0000000000FF00FF00FF00FF0000000000FF00FF000000
      0000FF00FF00FF00FF00FF00FF0000000000FF00FF0086806D0000000000D8CE
      B900BEB59800BEB5980086806D00544E3F00D8CEB90086806D00544E3F00D8CE
      B90000000000A99E750086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FF00FF0000000000FF00
      FF0000000000FF00FF0000000000FF00FF00FF00FF0000000000FF00FF000000
      0000FF00FF00FF00FF00FF00FF0000000000FF00FF0086806D0000000000D8CE
      B900D8CEB900BEB5980086806D00544E3F00D8CEB90086806D00544E3F00D8CE
      B90000000000A99E750086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FF00FF0000000000FF00
      FF0000000000FF00FF0000000000FF00FF00FF00FF0000000000FF00FF000000
      0000FF00FF00FF00FF00FF00FF0000000000FF00FF0086806D0000000000D8CE
      B900BEB59800BEB5980086806D0086806D0086806D0086806D0086806D008680
      6D00A99E7500A99E750086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FF00FF0000000000FF00
      FF0000000000FF00FF0000000000FF00FF00FF00FF0000000000FF00FF000000
      0000FF00FF00FF00FF00FF00FF0000000000FF00FF0086806D0000000000D8CE
      B9006B5ED200BEB59800BEB59800BEB59800BEB59800BEB59800BEB59800A99E
      7500A99E7500A99E750086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000086806D0000000000D8CE
      B9006B5ED2006B5ED200A99E7500A99E7500A99E7500A99E7500A99E7500A99E
      7500A99E7500A99E750086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000FF00000000000000000000000000
      0000000000000000000000000000FF000000000000000000000000000000FF00
      0000000000000000000000000000000000006B5ED2006B5ED2006B5ED2006B5E
      D2006B5ED2006B5ED2006B5ED200FFFBDE00FFFBDE00FFFBDE00FFFBDE00FFFB
      DE00FFFBDE00A99E750086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000FF00000000000000000000000000
      0000000000000000000000000000FF000000000000000000000000000000FF00
      00000000000000000000000000006B5ED2000000000086806D0000000000D8CE
      B9006B5ED2006B5ED200FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF
      9C00FFFF9C00A99E750086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000FF00000000000000000000000000
      0000000000000000000000000000FF000000000000000000000000000000FF00
      000000000000000000006B5ED200000000000000000086806D0000000000D8CE
      B9006B5ED200FFFBDE00FFFBDE00FFFBDE00FFFBDE00FFFBDE00FFFBDE00FFFB
      DE00FFFBDE00A99E750086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000FF00000000000000000000000000
      0000000000000000000000000000FF000000000000000000000000000000FF00
      000000000000000000006B5ED200000000000000000086806D0000000000D8CE
      B900FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF
      9C00FFFF9C00A99E750086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000FF00000000000000000000000000
      0000000000000000000000000000FF000000000000000000000000000000FF00
      000000000000000000006B5ED200000000000000000086806D0000000000D8CE
      B900000000000000000000000000000000000000000000000000000000000000
      000000000000544E3F0086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000FF00000000000000000000000000
      0000000000000000000000000000FF000000000000000000000000000000FF00
      0000000000000000000000000000000000000000000086806D0000000000D8CE
      B90086806D0086806D0086806D0086806D0086806D0086806D0086806D008680
      6D0086806D00BEB5980086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000FF00000000000000000000000000
      0000000000000000000000000000FF000000000000000000000000000000FF00
      0000000000000000000000000000000000000000000086806D0086806D008680
      6D0086806D0086806D0086806D0086806D0086806D0086806D0086806D008680
      6D0086806D0086806D0086806D00000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF000000000000000000000000000000FF00000000000000
      FF000000FF00000000000000FF0000000000000000000000FF00000000000000
      0000000000000000FF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000FF0000000000000000000000FF
      000000000000000000000000000000FF00000000000000FF0000000000000000
      0000000000000000000000FF0000000000000000000000000000000000000000
      000000FF00000000000000FF0000000000000000000000000000000000000000
      00000000000000FF000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000FF0000000000000000000000FF
      000000000000000000000000000000FF00000000000000FF0000000000000000
      0000000000000000000000FF0000000000000000000000000000000000000000
      000000FF00000000000000FF0000000000000000000000000000000000000000
      00000000000000FF000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000FF0000000000000000000000FF
      000000000000000000000000000000FF00000000000000FF0000000000000000
      0000000000000000000000FF0000000000000000000000000000000000000000
      000000FF00000000000000FF0000000000000000000000000000000000000000
      00000000000000FF000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000FF0000000000000000000000FF
      000000000000000000000000000000FF00000000000000FF0000000000000000
      0000000000000000000000FF0000000000000000000000000000000000000000
      000000FF00000000000000FF0000000000000000000000000000000000000000
      00000000000000FF000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000FF0000000000000000000000FF
      000000000000000000000000000000FF00000000000000FF0000000000000000
      0000000000000000000000FF0000000000000000000000000000000000000000
      000000FF00000000000000FF0000000000000000000000000000000000000000
      00000000000000FF000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000FF0000000000000000000000FF
      000000000000000000000000000000FF00000000000000FF0000000000000000
      0000000000000000000000FF0000000000000000000000000000000000000000
      000000FF00000000000000FF0000000000000000000000000000000000000000
      00000000000000FF000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000FF0000000000000000000000FF
      000000000000000000000000000000FF00000000000000FF0000000000000000
      0000000000000000000000FF0000000000000000000000000000000000000000
      000000FF00000000000000FF0000000000000000000000000000000000000000
      00000000000000FF000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000000000000000FFFF
      F700FBF7F700FBF7F700FBF7F700FFFBDE00FFFBDE00FBF7F700544E3F00544E
      3F00544E3F00544E3F00544E3F00544E3F00544E3F00544E3F00544E3F00544E
      3F00544E3F00544E3F00544E3F00000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000007B635A007B5A5A00636363007B5A5A007B635A006B6B6B006B6B
      6B00000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FFFFF700FFFBDE00F2ED
      D800ECDEB700ECDEB700ECDEB700ECDEB700ECDEB70086806D0086806D008680
      6D0086806D0086806D00544E3F00544E3F00544E3F00544E3F00544E3F008680
      6D0086806D0086806D0086806D00544E3F000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000ADB5CE004A638C0018315A0018315A004A638C00ADB5CE00000000000000
      0000000000002142420021424200214242002142420021424200214242002142
      4200214242002142420000000000000000000000000000000000000000000000
      00000000000000000000B5AD9C00000000000000000000000000000000000000
      0000635252007B635A007B6B6B00737373006B6B7B009484A5008C6363006B6B
      6B006B6B6B002142420021424200214242002142420021424200214242002142
      4200214242002142420000000000000000000000000000000000000000000000
      00000000000000000000B5AD9C00000000000000000000000000000000000000
      0000000000000000000000000000FBF7F700FFFBDE00F2EDD800E7DEAD00E7DE
      AD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD0086806D0000000000BEB5
      9800BEB59800BEB5980086806D0086806D0086806D0086806D00544E3F00D8CE
      B90000000000A99E750086806D00544E3F00000000000000000000000000FF99
      0000CC330000CC990000CC660000CC990000FFCC0000FF990000FFCC0000FF99
      0000CCCC6600CCFF33000033CC00000099000033FF000099FF0033CCFF0000FF
      FF0033FFCC0000CCFF000099FF000099FF000099FF000099FF000099FF000099
      FF000066FF000099FF000000000000000000000000000000000000000000ADB5
      CE00314A7300295AAD00397BDE00428CDE00396BAD00314A7300ADB5CE000000
      0000000000002142420073DEDE0073DEDE006BCED6006BCED6006BCED6006BCE
      D6006BCED6002142420000000000000000000000000000000000000000000000
      0000CEC6B5005A4A29007363420000000000000000000000000000000000ADB5
      CE007B635A006B524A00847B73006384AD00427BCE00526BAD009484A5008C63
      63006B6B6B006B6B6B0073DEDE0073DEDE006BCED6006BCED6006BCED6006BCE
      D6006BCED6002142420000000000000000000000000000000000000000000000
      0000CEC6B5005A4A290073634200000000000000000000000000000000000000
      00000000000000000000FBF7F700F2EDD800ECDEB700DED2A800A99E7500A99E
      7500BEB59800BEB59800A99E7500BEB59800DED2A80086806D0000000000D8CE
      B900BEB59800BEB5980086806D00544E3F00D8CEB90086806D00544E3F00D8CE
      B90000000000A99E750086806D00544E3F000000000000000000FF990000FF99
      0000FF330000FF990000FF990000FF990000FFCC0000FF990000FF990000FF99
      0000CCCC3300CCFF33000033CC00000099000033CC000066FF003399CC003399
      99003399CC0000CCFF000099FF000066FF000099FF000099FF000066FF000066
      FF000066FF000099FF0000000000000000000000000000000000000000004A63
      8C00295AAD00428CEF004A8CF700529CF7005AADF700396BAD005A738C000000
      0000000000002142420073DEDE0073DEDE0073DEDE0073DEDE006BCED6006BCE
      D6006BCED600214242000000000000000000000000000000000000000000947B
      6300735A3100BD944A0073634200000000000000000000000000000000004A63
      8C00635252007B635A005273940052BDFF00399CFF00427BCE007373A5009484
      A5007B635A006B6B6B006B6B6B0073DEDE0073DEDE0073DEDE006BCED6006BCE
      D6006BCED600214242000000000000000000000000000000000000000000947B
      6300735A3100BD944A0073634200000000000000000000000000000000000000
      000000000000FBF7F700ECDEB700E7DEAD00E7DEAD00DED2A800A99E7500A99E
      7500DED2A800DED2A80086806D00BEB59800DED2A80086806D0000000000D8CE
      B900D8CEB900BEB5980086806D00544E3F00D8CEB90086806D00544E3F00D8CE
      B90000000000A99E750086806D00544E3F00000000000000000000000000FF99
      0000CC330000FF99000000000000FF990000FFCC000000000000FF990000FF99
      0000FFCC0000FFCC00000066FF00000099000066FF000099FF003399990033CC
      99003399990033CCFF0033CCFF000099FF000099FF000099FF000099FF000099
      FF000099FF000099FF0000000000000000000000000000000000000000001831
      5A00397BDE00529CF7005AADF7006BB5F7006BB5F70063A5DE0029425A000000
      000000000000214242007BDEE7007BDEE70073DEDE0021424200214242002142
      42002142420021424200000000000000000000000000CEC6B5005A4A2900A584
      4200EFB56300BD944A0073634200000000000000000000000000000000001831
      5A00397BDE00635252006B6B6B0052A5E7005AC6FF0039A5FF00296BCE007373
      A5009484A5008C6363006B6B6B006B6B6B0073DEDE0021424200214242002142
      42002142420021424200000000000000000000000000CEC6B5005A4A2900A584
      4200EFB56300BD944A0073634200000000000000000000000000000000000000
      0000FBF7F700ECDEB700E7DEAD00E7DEAD00E7DEAD00E7DEAD00BEB59800BEB5
      9800E7DEAD00E7DEAD00A99E7500E7DEAD00E7DEAD0086806D0000000000D8CE
      B900BEB59800BEB5980086806D0086806D0086806D0086806D0086806D008680
      6D00A99E7500A99E750086806D00544E3F000000000000000000CC3300000000
      0000CC330000CC660000FF330000CC66000000000000FF330000CC330000CC66
      000000000000FFCC000033CCFF000033FF000099FF0000FFFF0066CC9900FFFF
      FF0066CC990066FFCC0033FFCC0000FFFF0033CCFF0033FFCC00339999003399
      99003399990033FFCC0000000000000000000000000000000000000000001831
      5A00427BD600529CF7006BB5F7007BC6F7007BC6F7006BADE70029425A000000
      00000000000029424A007BDEE7007BDEE7002142420000000000000000000000
      000000000000000000000000000000000000947B6300735A3100DEA55A00EFB5
      6300EFB56300BD944A0073634200000000000000000000000000000000001831
      5A00427BD600529CF7006B6B6B005A5A5A0052A5E70052BDFF00399CFF00427B
      CE00526BAD009484A5007B635A006B6B6B006B6B6B0000000000000000000000
      000000000000000000000000000000000000947B6300735A3100DEA55A00EFB5
      6300EFB56300BD944A007363420000000000000000000000000000000000FBF7
      F700ECDEB700E7DEAD00E7DEAD00E7DEAD00E7DEAD00DED2A80086806D00A99E
      7500E7DEAD00BEB5980086806D00E7DEAD00E7DEAD0086806D0000000000D8CE
      B9006B5ED200BEB59800BEB59800BEB59800BEB59800BEB59800BEB59800A99E
      7500A99E7500A99E750086806D00544E3F000000000000000000CC330000CC33
      0000CC330000CC330000FF000000CC330000CC330000CC000000FF000000CC33
      000099333300CC66330066CC99000099FF0033FFCC0066FF990099CC6600FFFF
      FF0099996600CCFF660066FF660066FF990099FF990066CC990066999900FFFF
      FF006699990099CC990000000000000000000000000000000000000000004A63
      8C00396BAD005AADF7006BB5F7007BC6F7008CCEFF005A8CB500637B9C000000
      000000000000294A4A0084E7EF0084E7EF0029424A0000000000000000000000
      00000000000000000000CEC6B5005A4A3100A5844200EFB56300EFB56300EFB5
      6300EFB56300BD944A0073634200000000000000000000000000000000004A63
      8C00396BAD005AADF7006BB5F7009C391000634A4A005AADF70052BDFF0039A5
      FF00427BCE007373A5009484A5008C6B7300737373006B6B6B00000000000000
      00000000000000000000CEC6B5005A4A3100A5844200EFB56300EFB56300EFB5
      6300EFB56300BD944A0073634200000000000000000000000000FBF7F700F2ED
      D800E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00DED2A800DED2
      A80000FF0000DED2A800DED2A800E7DEAD00E7DEAD0086806D0000000000D8CE
      B9006B5ED2006B5ED200A99E7500A99E7500A99E7500A99E7500A99E7500A99E
      7500A99E7500A99E750086806D00544E3F000000000000000000CC000000FF00
      0000CC330000CC330000CC000000CC000000CC000000CC000000FF000000CC33
      000099333300FFFFFF009999660066CC990066FF66009999330099993300FFFF
      FF0099663300FFCC3300FFCC3300CCCC660099FF990099CC330066996600FFFF
      FF0099996600999933000000000000000000000000000000000000000000ADB5
      CE00314A7300396BAD0063A5DE006BADE7005A8CB500425A7300B5C6D6000000
      000000000000294A4A0084E7EF0084E7EF00294A4A0000000000000000000000
      000000000000947B6300735A3100DEA55A00EFB56300EFB56300EFB56300EFB5
      6300EFB56300BD944A007363420000000000000000000000000000000000ADB5
      CE00314A7300396BAD0063A5DE006BADE700E75239007B5A5A0052A5E7005AC6
      FF0039A5FF00296BCE007373A5009484A500635252006B6B6B006B6B6B006B6B
      6B006B6B6B006B6B6B006B6B6B006B6B6B006B6B6B006B6B6B00EFB56300EFB5
      6300EFB56300BD944A0073634200000000000000000000000000FFFBDE00ECDE
      B700E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DE
      AD0000FF0000E7DEAD00E7DEAD00E7DEAD006B5ED2006B5ED2006B5ED2006B5E
      D2006B5ED2006B5ED2006B5ED200FFFBDE00FFFBDE00FFFBDE00FFFBDE00FFFB
      DE00FFFBDE00A99E750086806D00544E3F000000000000000000CC000000FF00
      0000996633009933330099663300CC000000CC330000CC00000000000000CC33
      000099333300FFFFFF00FFFFFF009999660066FF660099993300FFFFFF00FFFF
      FF0099663300CC99000000000000FFCC0000CCCC66009999660099996600FFFF
      FF0099996600CC99330000000000000000000000000000000000000000000000
      0000ADB5CE005A738C0029425A0029425A00637B9C00B5C6D600000000000000
      000000000000294A4A0084E7EF0084E7EF00294A4A000000000000000000CEC6
      B5005A4A3100A5844A00CEA55A00CEA55A00CE9C5200CE9C5200CE9C5200CE9C
      5200CE9C5200BD944A0073634200000000000000000000000000000000000000
      0000ADB5CE005A738C0029425A0029425A00637B9C00C67352007B524A0052A5
      E7005AC6FF0039A5FF00296BCE007373A5007B635A00634A4A006B6B6B007B5A
      5A00635252007B524A007B635A007B635A007B635A006B6B6B006B6B6B006B6B
      6B00CE9C5200BD944A0073634200000000006B5ED2006B5ED2006B5ED2006B5E
      D2006B5ED2006B5ED2006B5ED2006B5ED2006B5ED2006B5ED2006B5ED2006B5E
      D20000FF0000E7DEAD00E7DEAD006B5ED200E7DEAD0086806D0000000000D8CE
      B9006B5ED2006B5ED200FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF
      9C00FFFF9C00A99E750086806D00544E3F00000000000000000099330000FF33
      000099663300FFFFFF00CC663300000000000000000099330000FF9900000000
      000099663300FFFFFF00FFFFFF009999660066FFCC0099996600FFFFFF00FFFF
      FF0099996600CC990000FF330000FF990000CCCC3300CC996600FFFFFF00FFFF
      FF0099996600FFCC33000000000000000000000000003939210073735A000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000294A4A00294A4A00294A4A00294A4A0000000000B5AD9C006B5A
      42006B5A42006B5A39006B5A39006B5A39006B5A39006B5A39006B5A39006B5A
      39006B5A39006B5A3900947B630000000000000000003939210073735A000000
      0000000000000000000000000000000000000000000000000000AD523100847B
      730052A5E7005AC6FF005AADF7006384AD00737373007B524A007B524A00A573
      6B00C68C8C00D6AD9C00EFCEAD00E7CEBD00D6AD9C009C8473007B524A006B6B
      6B006B6B6B006B5A3900947B63000000000000000000FFFBDE00E7DEAD00E7DE
      AD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00DED2A800BEB5980000FF
      000000FF0000DED2A8006B5ED200BEB59800E7DEAD0086806D0000000000D8CE
      B9006B5ED200FFFBDE00FFFBDE00FFFBDE00FFFBDE00FFFBDE00FFFBDE00FFFB
      DE00FFFBDE00A99E750086806D00544E3F000000000000000000CC9900000000
      000099333300FFFFFF00CC993300FF990000FF990000CC660000FFCC3300FF99
      0000CC993300FFFFFF00FFFFFF009999330033FFCC0066999900FFFFFF00FFFF
      FF0066996600CCCC000000000000FF990000FF99330099993300FFFFFF00FFFF
      FF0099996600CC993300000000000000000000000000313121006B7342004242
      21005A5A42000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000313121006B7342004242
      21005A5A4200000000000000000000000000000000000000000000000000D6AD
      9C007B635A004A9CDE0063A5CE009C9C9C008C8C8C00B58C8400CEA58C00EFCE
      AD00FFEFC600FFFFCE00FFFFCE00FFFFD600FFFFD600FFF7D600DEBDAD007B63
      5A007B6B6B006B6B6B000000000000000000FFFFF700F2EDD800E7DEAD006B5E
      D200E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00DED2A800A99E750000FF
      0000DED2A80000FF00006B5ED200A99E7500DED2A80086806D0000000000D8CE
      B900FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF9C00FFFF
      9C00FFFF9C00A99E750086806D00544E3F000000000000000000FF9900000000
      000099663300FFFFFF0099993300FF990000FF9900000000000099993300CC66
      3300CC993300FFFFFF00FFFFFF009999660066CC990066996600FFFFFF00FFFF
      FF0099996600CCCC330000000000CC990000FF99000099663300FFFFFF00FFFF
      FF0099666600FF990000000000000000000000000000313121009CA55A009CA5
      5A00313118000000000000000000183942001839420010313900103139001029
      3900102939000000000000000000000000000000000000000000000000000000
      0000B5CEC600638473003152420018392900183929003152420063847300B5CE
      C6000000000000000000000000000000000000000000313121009CA55A009CA5
      5A00313118000000000000000000183942001839420010313900103139001029
      3900C6735200B56342009C5A52009C847300C68C8C00E7BD9C00FFE7B500FFFF
      CE00FFFFCE00FFFFCE00FFFFD600FFFFDE00FFFFDE000000000000000000E7D6
      BD007B5A5A00737373006B6B6B0000000000FBF7F700ECDEB7006B5ED2006B5E
      D2006B5ED200E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00BEB5980000FF
      0000E7DEAD0000FF00006B5ED200A99E7500DED2A80086806D0000000000D8CE
      B900000000000000000000000000000000000000000000000000000000000000
      000000000000544E3F0086806D00544E3F000000000000000000CC6633009966
      330099663300FFFFFF009999330099663300FF9900000000000099993300FFFF
      FF00CC663300FFFFFF0099993300FFFFFF009999660099996600FFFFFF00FFFF
      FF009999660099993300CC990000CCCC3300FF990000CC663300FFFFFF00FFFF
      FF00996633000000000000000000000000000000000031312100A5A563009CA5
      5A00313121000000000000000000183942005AC6DE0052BDD6004AADCE004AAD
      CE00102939000000000000000000000000000000000000000000000000007B9C
      8C00294A39002963420031845A004A9C6B004A9C6B00428C6300316B52003152
      4200849C8C000000000000000000000000000000000031312100A5A563009CA5
      5A00313121000000000000000000183942005AC6DE0052BDD6004AADCE004AAD
      CE0010293900CE5A29009C5A52009C6B6B00E7BD9C00FFE7BD00FFEFC600FFEF
      BD00FFFFCE00FFFFD600FFFFDE00000000000000000000000000000000000000
      0000DEBDAD005A4242006B6B6B0000000000FBF7F700ECDEB700E7DEAD006B5E
      D200E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00DED2A800A99E750000FF
      0000E7DEAD0000FF0000A99E7500A99E7500DED2A80086806D0000000000D8CE
      B90086806D0086806D0086806D0086806D0086806D0086806D0086806D008680
      6D0086806D00BEB5980086806D00544E3F00000000000000000099663300FFFF
      FF0099333300FFFFFF00FFFFFF0099663300FF990000FF99000099993300FFFF
      FF00FFFFFF00FFFFFF0099993300FFFFFF009999660099993300FFFFFF009999
      6600FFFFFF0099993300CC990000CCCC3300FFCC3300CC996600FFFFFF00FFFF
      FF0099663300CC33000000000000000000000000000031312100A5AD6300A5A5
      6300313121000000000000000000214242001839420018394200183942001031
      39001031390000000000000000000000000000000000000000007B9C8C00215A
      420031845A004AA573004AA5730052AD7B005AAD84005AAD7B0063AD84004A9C
      6B0031634A00849C8C0000000000000000000000000031312100A5AD6300A5A5
      6300313121000000000000000000214242001839420018394200183942001031
      390010313900CE8C7300945A4A00CEA58C00FFDEAD00FFF7C600FFE7B500FFEF
      BD00FFFFCE00FFFFD600FFFFDE00000000000000000000000000000000000000
      0000FFFFD600AD7B7300636363006B6B6B00FBF7F700ECDEB700E7DEAD006B5E
      D200E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00BEB5980000FF
      0000E7DEAD0000FF0000BEB59800DED2A800E7DEAD0086806D0086806D008680
      6D0086806D0086806D0086806D0086806D0086806D0086806D0086806D008680
      6D0086806D0086806D0086806D00FBF7F7000000000000000000CC663300FFFF
      FF0099333300FFFFFF00FFFFFF0099663300CC993300CC99330099993300FFFF
      FF0099663300FFFFFF0099993300FFFFFF009999330099996600FFFFFF009999
      6600FFFFFF009999330099993300999966009999660099CC6600FFFFFF00FFFF
      FF00999933009966330000000000000000000000000031312100A5AD6300A5A5
      6300313121000000000000000000000000000000000000000000000000000000
      00000000000000000000735A73000000000000000000B5CEC600294A39003184
      5A004AA5730052AD7B005AAD7B0063AD840063AD840063AD84006BBD8C0063AD
      84005A9C730031524200BDCEC600000000000000000031312100A5AD6300A5A5
      6300313121000000000000000000000000000000000000000000000000000000
      000000000000D69C84009C6B6B00E7BD9C00FFF7CE00FFEFBD00FFDEAD00FFEF
      BD00FFFFCE00FFFFD600FFFFDE0000000000000000000000000000000000FFFF
      DE00FFFFDE00DEBDAD00424242006B6B6B00FFFBDE00ECDEB700E7DEAD006B5E
      D200E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD0000FF
      0000E7DEAD0000FF0000E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DE
      AD00E7DEAD00E7DEAD00E7DEAD00E7DEAD0000FF0000E7DEAD00E7DEAD00E7DE
      AD00E7DEAD00E7DEAD00ECDEB700FBF7F7000000000000000000CC993300FFFF
      FF0099663300FFFFFF00FFFFFF0099993300FFFFFF00FFFFFF0099996600FFFF
      FF00996666009999330099996600FFFFFF009999330099996600FFFFFF00CC99
      6600FFFFFF009999330099663300FFFFFF009999330099CC6600FFFFFF009999
      3300FFFFFF009999330000000000000000000000000031312100ADAD6B00A5AD
      63004A4A2900CECEBD00CECEBD00CECEBD00CECEBD00CECEBD00CECEBD000000
      000000000000735A73006B4A6B00000000000000000063847300215A42004AA5
      73004AA573005AAD7B0063AD84006BBD8C006BBD8C0073BD8C0073BD8C0073BD
      8C0073BD8C004A735A00738C7B00000000000000000031312100ADAD6B00A5AD
      63004A4A2900CECEBD00CECEBD00CECEBD00CECEBD00CECEBD00CECEBD000000
      000000000000BD9C9400A5737300EFCEAD00FFF7C600FFDEAD00FFDEA500FFEF
      BD00FFFFCE00FFFFD600FFFFDE0000000000000000000000000000000000FFFF
      DE00FFFFE700EFCEAD005A4242006B6B6B00FFFBDE00ECDEB700E7DEAD006B5E
      D200E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD0000FF0000E7DE
      AD00E7DEAD0000FF0000E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DE
      AD00E7DEAD00E7DEAD00E7DEAD00E7DEAD0000FF0000E7DEAD00E7DEAD00E7DE
      AD00E7DEAD00E7DEAD00ECDEB700FBF7F7000000000000000000CC993300FFFF
      FF0099993300FFFFFF00FFFFFF0099996600FFFFFF00FFFFFF0099996600FFFF
      FF00999966009999660066996600FFFFFF009999330099993300FFFFFF00CC99
      3300FFFFFF00FFFFFF0099663300FFFFFF00FFFFFF0099996600FFFFFF009999
      3300FFFFFF00CC99660000000000000000000000000031312100ADAD6B00ADAD
      6B00A5AD63004A4A29004A4A29004242290042422100424221008C8C73000000
      0000735A7300B57BAD006B4A6B0000000000000000003152420031845A004AA5
      730052AD7B0063AD840063AD840073BD8C007BC6940084C6940084C694007BC6
      94007BC694005A9C7300425A4A00000000000000000031312100ADAD6B00ADAD
      6B00A5AD63004A4A29004A4A29004242290042422100424221008C8C73000000
      0000735A7300B57BAD00AD7B7300FFE7BD00FFF7C600FFD6A500FFD6A500FFE7
      B500FFF7C600FFFFCE00FFFFD600FFFFDE00FFFFDE00FFFFE700FFFFDE00FFFF
      D600FFFFD600E7D6BD006B524A006B6B6B0000FF000000FF000000FF000000FF
      000000FF0000D8CEB900D8CEB900D8CEB900D8CEB900D8CEB90000FF0000D8CE
      B900D8CEB900D8CEB90000FF0000D8CEB90000FF000000FF000000FF000000FF
      000000FF000000FF0000D8CEB90000FF0000D8CEB90000FF0000BEB59800BEB5
      9800BEB59800BEB59800BEB59800BEB59800000000000000000099993300FFFF
      FF00FFFFFF00FFFFFF00FFFFFF00FFFFFF0099996600FFFFFF00FFFFFF00FFFF
      FF006699660099CC660066999900FFFFFF00CC99660099996600FFFFFF009999
      330099996600FFFFFF0099663300FFFFFF00FFFFFF0099996600FFFFFF009999
      330099993300FFFFFF0000000000000000000000000039392100B5B57300B5B5
      7300ADAD6B00A5A56300A5A563009CA55A005A5A31009C9C8C0000000000735A
      7300B57BAD00DE9CDE006B4A6B0000000000000000001839290042946B004AA5
      73005AAD7B0063AD840073BD8C007BC6940084C6940084C69C0084C69C0084C6
      9C0084C6940073B58C0029423100000000000000000039392100B5B57300B5B5
      7300ADAD6B00A5A56300A5A563009CA55A005A5A31009C9C8C0000000000735A
      7300B57BAD00DE9CDE00A5736B00FFE7B500FFF7C600FFDEAD00FFDEAD00FFE7
      B500FFEFBD00FFFFC600FFFFCE00FFFFD600FFFFDE00FFFFDE00FFFFD600FFFF
      CE00FFFFD600EFCEAD00634A4A006B6B6B00FBF7F700ECDEB700E7DEAD006B5E
      D200E7DEAD0000FF0000E7DEAD00E7DEAD00E7DEAD00DED2A80000FF00008680
      6D00BEB59800BEB5980000FF000000FF0000DED2A800A99E7500A99E7500BEB5
      9800BEB5980000FF0000A99E750000FF000086806D00BEB59800DED2A8008680
      6D00A99E7500DED2A800FFFBDE0000000000000000000000000099996600FFFF
      FF00FFFFFF00FFFFFF00FFFFFF00FFFFFF0099CC6600FFFFFF00FFFFFF00FFFF
      FF0099CC660066CC990066996600FFFFFF0099996600FFFFFF00999966009999
      330099996600FFFFFF0099663300FFFFFF00FFFFFF0099996600FFFFFF009999
      330099996600FFFFFF0000000000000000000000000039392900B5B57B00B5B5
      7300ADAD6B00ADAD6B00A5A563005A5A31009C9C8C0000000000735A7300B57B
      AD00DE9CDE00DE9CDE006B4A6B0000000000000000001839290042946B0052AD
      7B005AAD840063B5840073BD8C0084C6940084C69C0094CEA50094CEA50094CE
      A50084C69C007BBD8C0029423100000000000000000039392900B5B57B00B5B5
      7300ADAD6B00ADAD6B00A5A563005A5A31009C9C8C0000000000735A7300B57B
      AD00DE9CDE00DE9CDE008C6B7300EFCEAD00FFFFCE00FFEFC600FFE7BD00FFD6
      A500FFDEB500FFEFBD00FFF7C600FFFFCE00FFFFCE00FFFFCE00FFFFC600FFFF
      CE00FFFFCE00E7BD9C00424242006B6B6B00FBF7F700ECDEB700E7DEAD006B5E
      D200E7DEAD0000FF0000E7DEAD00E7DEAD00E7DEAD00E7DEAD0000FF0000BEB5
      9800E7DEAD00A99E750000FF0000A99E7500DED2A800A99E7500BEB59800A99E
      7500BEB5980000FF0000A99E750000FF0000A99E7500E7DEAD00E7DEAD00A99E
      7500BEB59800BEB59800FFFBDE0000000000000000000000000099996600FFFF
      FF00FFFFFF00FFFFFF00FFFFFF00FFFFFF0099CC6600FFFFFF0033CC9900FFFF
      FF009999660099CC990066999900FFFFFF0099996600FFFFFF0099996600CCCC
      000099996600FFFFFF0099663300FFFFFF00CC993300FFFFFF00999966009999
      330099996600FFFFFF0000000000000000000000000039392900B5B57300B5B5
      7300B5B57300ADAD6B0052523100B5AD9C0000000000735A6B00B57BAD00DE9C
      DE00DE9CDE00DE9CDE006B4A6B0000000000000000003152420031845A0052AD
      7B005AAD840063AD84007BC6940084C6940084C69C0094CEA50094CEA50094CE
      A50084C69C006BA57B0042635200000000000000000039392900B5B57300B5B5
      7300B5B57300ADAD6B0052523100B5AD9C0000000000735A6B00B57BAD00DE9C
      DE00DE9CDE00DE9CDE006B4A6B00D6AD9C00FFF7CE0000000000FFF7D600FFEF
      C600FFDEAD00FFE7B500FFEFBD00FFEFBD00FFEFBD00FFEFBD00FFEFBD00FFFF
      CE00FFEFBD00C68C8C00525252006B6B6B00FFFFF700F2EDD800E7DEAD006B5E
      D200E7DEAD00E7DEAD0000FF0000E7DEAD00E7DEAD00DED2A80000FF0000BEB5
      9800E7DEAD00BEB59800A99E7500A99E7500DED2A800A99E7500A99E7500BEB5
      9800BEB5980000FF0000A99E750000FF000086806D00E7DEAD00E7DEAD008680
      6D00A99E7500BEB59800FBF7F70000000000000000000000000099996600FFFF
      FF00FFFFFF00FFFFFF00FFFFFF00FFFFFF00999966006699990066CC9900FFFF
      FF009999330099CC660099996600FFFFFF0099666600FFFFFF0099993300CCCC
      330099996600FFFFFF0099663300FFFFFF00CC993300FFFFFF0099996600FFCC
      000099996600CC99330000000000000000000000000039392900BDBD7B00BDBD
      7B00B5B573005A5A3100BDBDAD0000000000735A7300B57BAD00DE9CDE00DE9C
      DE00DE9CDE00DE9CDE006B4A6B000000000000000000638473002963420052AD
      7B0063AD840063AD840073BD8C0084C6940084C69C0094CEA50094CEA50094CE
      A50084C69C00527B5A00738C7B00000000000000000039392900BDBD7B00BDBD
      7B00B5B573005A5A3100BDBDAD0000000000735A7300B57BAD00DE9CDE00DE9C
      DE00DE9CDE00DE9CDE006B4A6B00AD7B7300EFE7BD0000000000000000000000
      0000FFE7BD00FFD6A500FFDEB500FFDEA500FFDEAD00FFDEAD00FFF7C600FFF7
      C600F7CE9C006B524A006B6B6B000000000000000000FFFBDE00E7DEAD006B5E
      D200E7DEAD00E7DEAD0000FF0000E7DEAD00E7DEAD00DED2A80000FF0000DED2
      A800E7DEAD00DED2A800BEB59800BEB59800E7DEAD00BEB59800BEB59800DED2
      A800DED2A800BEB5980000FF000000FF0000BEB59800E7DEAD00E7DEAD00BEB5
      9800BEB59800E7DEAD00FBF7F700000000000000000000000000FFFFFF006699
      6600FFFFFF009999660099996600FFFFFF006699660099FF660099CC66006699
      660099993300CCFF330099996600FFFFFF00FFFFFF00FFFFFF0099996600CCFF
      66009999660099663300FFFFFF00FFFFFF00CC663300FFFFFF0099993300FF99
      0000CC993300FFCC000000000000000000000000000039392900BDBD7B00B5B5
      730052523100C6C6B50000000000735A7300734A6B00845A7B00845A7B00845A
      7B00845A7B00845A7B006B4A6B000000000000000000BDCEC600294A39004294
      6B005AAD7B0063AD84006BBD8C007BC6940084C6940084C69C0084C69C0084C6
      9C0073AD8400425A4A00BDCEC600000000000000000039392900BDBD7B00B5B5
      730052523100C6C6B50000000000735A7300734A6B00845A7B00845A7B00845A
      7B00845A7B00845A7B006B4A6B00634A4A00CE9C940000000000000000000000
      0000FFF7D600FFE7BD00FFD6A500FFD6A500FFE7B500FFF7C600FFF7C600FFD6
      A500B5847B005A5A5A00BDCEC6000000000000000000FFFFF700F2EDD8006B5E
      D200E7DEAD00E7DEAD0000FF0000E7DEAD00E7DEAD0000FF0000E7DEAD00E7DE
      AD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DE
      AD00E7DEAD00E7DEAD0000FF0000E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DE
      AD00E7DEAD00F2EDD800FFFFF700000000000000000000000000FFFFFF006699
      6600999966006699660099996600FFFFFF006699990099FF660099FF660099FF
      9900CCCC3300CCFF330099996600FFFFFF0066996600FFFFFF0099CC6600CCFF
      660099CC660099663300FFFFFF0099663300CC663300FFFFFF0099993300FF99
      0000CCCC6600FFCC000000000000000000000000000039392900B5B57B005252
      3100CECEBD000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000849C8C002963
      420042946B0063AD84006BBD8C0073BD8C007BC6940084C6940084C6940073AD
      84004A735A008CA5940000000000000000000000000039392900B5B57B005252
      3100CECEBD000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000007B524A00CE9C9400FFF7D6000000
      000000000000FFFFCE00FFEFBD00FFF7C600FFF7C600FFEFBD00F7CE9C00C68C
      8C005A524A008CA5940000000000000000000000000000000000FFFBDE006B5E
      D200E7DEAD00E7DEAD0000FF0000E7DEAD00E7DEAD0000FF0000E7DEAD00E7DE
      AD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DE
      AD00E7DEAD00E7DEAD0000FF0000E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DE
      AD00ECDEB700FFFBDE0000000000000000000000000000000000669999006699
      660099CC660033CC990099996600FFFFFF006699990099CC660099FF990066FF
      9900CCCC6600CCFF3300999966009999660066996600FFFFFF0099CC660099FF
      660099FF990099993300FFFFFF0099663300CC663300FFFFFF00999933000000
      000099CC6600FFCC0000000000000000000000000000393929004A4A3900D6D6
      CE00000000000000000000000000000000000000000000000000312942003129
      420031294200312142003121420029213900292139002921390000000000849C
      8C0031524200316B520042946B0063AD840063AD84005A9C7300527B5A00425A
      4A008CA5940000000000000000000000000000000000393929004A4A3900D6D6
      CE00000000000000000000000000000000000000000000000000312942003129
      420031294200312142003121420029213900292139007B635A00A5737300D6AD
      9C00EFCEAD00FFE7B500FFE7B500FFE7B500EFCEAD00E7BD9C009C6B6B005A5A
      5A008CA594000000000000000000000000000000000000000000FBF7F7006B5E
      D200E7DEAD00E7DEAD0000FF0000E7DEAD00E7DEAD0000FF0000DED2A800DED2
      A800E7DEAD00DED2A800DED2A800DED2A800DED2A800DED2A800DED2A800DED2
      A800DED2A800DED2A80000FF0000DED2A800DED2A800DED2A800E7DEAD00ECDE
      B700FBF7F700000000000000000000000000000000000000000066FF990099CC
      660099CC660033CC990099996600FFFFFF0066999900CCCC330099FF990066FF
      990099993300FFCC3300CC993300CCFF660066996600FFFFFF009999660066FF
      990066FF990099996600996633009966330099666600FFFFFF00CC993300CC99
      000099CC9900FFCC33000000000000000000000000004A4A3900DEDED6000000
      0000636B9C002129630021296300636B9C00000000000000000039314A00AD8C
      E700A584DE00A584DE009C7BD6009C7BD6009C7BD60029213900000000000000
      0000BDCEC600738C7B00425A4A002942310029423100425A4A00738C7B00BDCE
      C600000000009C9C8C004A42290000000000000000004A4A3900DEDED6000000
      0000636B9C002129630021296300636B9C00000000000000000039314A00AD8C
      E700A584DE00A584DE009C7BD6009C7BD6009C7BD600292139007B635A007B63
      5A007B635A009C6B6B00B5847B00B58C84008C6B73006B63630063636300BDCE
      C600000000009C9C8C004A422900000000000000000000000000000000006B5E
      D200ECDEB700E7DEAD00E7DEAD0000FF0000E7DEAD0000FF000086806D008680
      6D00DED2A800BEB59800544E3F00A99E7500DED2A800A99E750086806D00BEB5
      9800BEB5980086806D00A99E7500A99E7500A99E7500A99E7500DED2A800F2ED
      D80000000000000000000000000000000000000000000000000099FF6600CCCC
      660099CC330066CC990099993300FFFFFF0066CC9900CC990000CCFF330099CC
      660099993300FFCC0000FF990000CCCC330066996600999966009999660066FF
      990066FF9900CCFF6600CC660000CC9900009966660099993300CC993300CC99
      000099CC9900CCCC00000000000000000000000000000000000000000000636B
      9C0029529C003963D6003963D60029399C00636B9C000000000031294200B594
      E700AD8CE700A584DE00A584DE009C7BD6009C7BD6002921390000000000C6BD
      AD0000000000000000000000000000000000000000000000000000000000D6D6
      CE005A5231007B6339006352310000000000000000000000000000000000636B
      9C0029529C003963D6003963D60029399C00636B9C000000000031294200B594
      E700AD8CE700A584DE00A584DE009C7BD6009C7BD6002921390000000000C6BD
      AD009C6B6B009C6B6B009C6B6B009C6B6B009C6B6B000000000000000000D6D6
      CE005A5231007B633900635231000000000000000000000000006B5ED2006B5E
      D2006B5ED200ECDEB700E7DEAD0000FF0000E7DEAD0000FF0000A99E7500BEB5
      9800BEB59800E7DEAD00A99E7500E7DEAD00E7DEAD00A99E7500BEB59800BEB5
      9800E7DEAD00A99E7500DED2A800A99E7500BEB59800A99E7500D8CEB900FBF7
      F700000000000000000000000000000000000000000000000000CCCC3300CC99
      0000CC990000CCFF6600999933009966330099996600CC990000FFCC0000CCCC
      3300CC990000FF990000CC660000FFCC0000CCFF3300FFCC330099CC660033FF
      990099FF990099FF6600CCCC0000CCCC6600CC990000CCCC0000CCCC3300CCCC
      000099FF990099FF660000000000000000000000000000000000000000002939
      6300427BD600529CF7004A8CF7003963D600212963000000000039314A00B594
      E700B594E700AD8CE700A584DE00A584DE009C7BD60031214200000000003129
      1000524A2900BDBDAD000000000000000000000000000000000084735A004A42
      2100A5844A00CEA55A0063523100000000000000000000000000000000002939
      6300427BD600529CF7004A8CF7003963D600212963000000000039314A00B594
      E700B594E700AD8CE700A584DE00A584DE009C7BD60031214200000000003129
      1000524A2900BDBDAD000000000000000000000000000000000084735A004A42
      2100A5844A00CEA55A0063523100000000000000000000000000000000006B5E
      D20000000000FBF7F700ECDEB70000FF0000E7DEAD0000FF0000A99E7500A99E
      7500BEB59800DED2A80086806D00E7DEAD00E7DEAD00A99E7500A99E7500BEB5
      9800DED2A80086806D00DED2A800A99E7500BEB59800D8CEB900FBF7F7000000
      0000000000000000000000000000000000000000000000000000FF993300FF99
      0000CC660000FFCC3300996633009966330099993300CC66000000000000CC99
      0000CC66000000000000CC330000CC660000CCCC3300FF993300CCCC330066FF
      660099FF660099FF6600CCCC3300CCCC3300CCCC3300CCCC330099FF6600CCCC
      330099FF990099FF990000000000000000000000000000000000000000002939
      6300428CDE005AADF700529CF7003963D600212963000000000039314A00B594
      E700B594E700AD8CE700AD8CE700A584DE00A584DE0031214200000000003129
      10007B63210042391800524A2900BDBDAD00B5AD9C00423918007B633100B594
      4A00BD9C5200BD9C520063523100000000000000000000000000000000002939
      6300428CDE005AADF700529CF7003963D600212963000000000039314A00B594
      E700B594E700AD8CE700AD8CE700A584DE00A584DE0031214200000000003129
      10007B63210042391800524A2900BDBDAD00B5AD9C00423918007B633100B594
      4A00BD9C5200BD9C520063523100000000000000000000000000000000000000
      00000000000000000000FBF7F70000FF000000FF0000DED2A800BEB59800BEB5
      9800DED2A800DED2A800A99E7500E7DEAD00E7DEAD00BEB59800BEB59800DED2
      A800DED2A800A99E7500DED2A800DED2A800D8CEB900FBF7F700000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000CC330000FF99000000000000CC330000CC660000CC330000CC330000CC33
      0000CC330000FF330000CC330000CC330000CC660000CC660000FFCC0000CCCC
      3300CCFF3300CCCC660099CC330099CC3300CCFF6600CCCC660066FF990099FF
      660099FF660066FFCC000000000000000000000000000000000000000000637B
      9C0029529C00428CDE00427BD60029529C00636B9C000000000039314A00BD9C
      EF00BD9CEF00B594E700B594E700AD8CE700A584DE0031294200000000003931
      10007B6321009C7B29007B63210042391800524218009C7B2900AD8C4200AD8C
      4200B5944A00BD9C52005A4A290000000000000000000000000000000000637B
      9C0029529C00428CDE00427BD60029529C00636B9C000000000039314A00BD9C
      EF00BD9CEF00B594E700B594E700AD8CE700A584DE0031294200000000003931
      10007B6321009C7B29007B63210042391800524218009C7B2900AD8C4200AD8C
      4200B5944A00BD9C52005A4A2900000000006B5ED2006B5ED2006B5ED2006B5E
      D2006B5ED2006B5ED2006B5ED2006B5ED20000FF00006B5ED2006B5ED2006B5E
      D2006B5ED2006B5ED200E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DEAD00E7DE
      AD00E7DEAD00E7DEAD00F2EDD800FFFBDE00FFFFF70000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000CC33000000000000CC660000CC330000FF330000FF330000CC000000FF33
      0000FF000000CC330000FF000000CC000000FF33000099330000FF990000FF99
      3300FFCC0000CC993300CC993300CCCC330099FF6600CCCC660066CC990099FF
      660099FF660066FFCC0000000000000000000000000000000000000000000000
      0000637B9C002939630029396300636B9C00000000000000000039314A00BD9C
      EF00BD9CEF00B594E700B594E700AD8CE700AD8CE70031294200000000004239
      18007B5A10008C6B18009C7B29009C7B29009C7B2900A5843900A5843900AD8C
      4200AD8C4200B5944A005A4A2900000000000000000000000000000000000000
      0000637B9C002939630029396300636B9C00000000000000000039314A00BD9C
      EF00BD9CEF00B594E700B594E700AD8CE700AD8CE70031294200000000004239
      18007B5A10008C6B18009C7B29009C7B29009C7B2900A5843900A5843900AD8C
      4200AD8C4200B5944A005A4A2900000000000000000000000000000000000000
      00000000000000000000000000000000000000FF0000FFFFF700FFFBDE00F2ED
      D800ECDEB700ECDEB700ECDEB700ECDEB700ECDEB700ECDEB700ECDEB700ECDE
      B700F2EDD800FFFBDE00FFFFF700000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000000039314A003931
      4A0039314A0039314A003129420039314A003129420031294200000000004239
      1800292108002921080029210800292108002921080031291000312910003129
      1000312910003129100031291000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000000039314A003931
      4A0039314A0039314A003129420039314A003129420031294200000000004239
      1800312910003129100031291000312910003129100031291000312910003129
      1000312910003129100031291000000000000000000000000000000000000000
      00000000000000000000000000000000000000FF00000000000000000000FFFF
      F700FBF7F700FBF7F700FBF7F700FFFBDE00FFFBDE00FBF7F700FBF7F700FBF7
      F700FFFFF7000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000424D3E000000000000003E000000
      2800000080000000600000000100010000000000000600000000000000000000
      000000000000000000000000FFFFFF0000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000FFFFFFFF000000000000000000000000
      AA514001000000000000000000000000AA510000000000000000000000000000
      AA512008000000000000000000000000AA512008000000000000000000000000
      AA512008000000000000000000000000AA512000000000000000000000000000
      AA512000000000000000000000000000FFFFA000000000000000000000000000
      7EEF00000000000000000000000000007EEEA000000000000000000000000000
      7EEDA0000000000000000000000000007EEDA000000000000000000000000000
      7EEDAFF80000000000000000000000007EEFA000000000000000000000000000
      7EEF8001000000000000000000000000FFFFFFFF000000000000000000000000
      A5BBA5BB000000000000000000000000A5BBA5BB000000000000000000000000
      A5BBA5BB000000000000000000000000A5BBA5BB000000000000000000000000
      A5BBA5BB000000000000000000000000A5BBA5BB000000000000000000000000
      A5BBA5BB000000000000000000000000FFFFFFFF000000000000000000000000
      6EBDF5FB0000000000000000000000006EBDF5FB000000000000000000000000
      6EBDF5FB0000000000000000000000006EBDF5FB000000000000000000000000
      6EBDF5FB0000000000000000000000006EBDF5FB000000000000000000000000
      6EBDF5FB000000000000000000000000FFE00001FFFFFFFFFFFFFFFFF80FFFFF
      FF800000FFFFFFFFF03803FDF00003FDFE002008E0000003E01803F1E00003F1
      FC002008C0000003E01803E1E00003E1F8002008E2400003E0180381E0000381
      F0002000D0880003E0187F01E0007F01E0002000C0000003E0187C01E0003C01
      C0002000C0000003E0187801E0000001C0000000C0200203F0386001F0000001
      00002000C19000039FF840019FC0000180002000D000020387FFFFFF87E00003
      00002000D04002038607F00F8600006100002FF8C04000078607E007860001F1
      00002000C00000038607C003860001F000000000C000000387FD800187F801E0
      00000000C000000380198001801801E000000000C00000038011800180100000
      00000000C0000003802180018020000000000001C00000038041800180400000
      00000001C0000003808180018080400000000001C00000038101800181007001
      80000001C0000003820180018200700180000001C000000387FFC00387FF1803
      C0000003C00000138FC020078FC00007C0000007C000000390C0300990C00009
      E000000FC0000003E0402FE1E0402061C000000FC0000003E04023C1E04023C1
      E800001FC0240003E0402001E0402001FC00003FF2000003E0402001E0402001
      0000007FF4000003F0C02001F0C02001FF0001FFFFFFFFFFFFC02001FFC02001
      FF6007FFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000000000000000000
      000000000000}
  end
  object DisplayToggleImages: TImageList
    Height = 13
    Width = 13
    Left = 442
    Top = 30
    Bitmap = {
      494C01010400090004000D000D00FFFFFFFFFF10FFFFFFFFFFFFFFFF424D3600
      000000000000360000002800000034000000270000000100200000000000B01F
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000000000000000FF000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000008080
      8000000000000000000000000000808080000000000000000000000000000000
      00000000000000000000000000000000000000000000000000000000000000FF
      000000000000000000000000FF00000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000FF00000000000000FF00000000
      00000000000000FF00000000000000FF0000000000000000000000FF00000000
      0000000000000000000000000000808080000000000000000000000000008080
      8000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000FF00000000000000FF00000000FF00000000000000
      FF00000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000FF00000000000000FF0000000000000000000000FF00000000000000FF
      0000000000000000000000FF0000000000000000000000000000808080000000
      0000000000000000000000000000000000008080800000000000000000000000
      00000000FF000000FF000000FF000000FF000000FF000000FF000000FF000000
      000000FF00000000FF00000000000000FF000000FF0000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000FF00000000000000FF00000000
      00000000000000FF00000000000000FF0000000000000000000000FF00000000
      0000000000000000000080808000000000000000000000000000000000000000
      00008080800000000000000000000000000000FF000000FF0000000000000000
      00000000000000FF00000000FF0000000000000000000000FF0000FF000000FF
      000000FF00000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000FF00000000000000FF0000000000000000000000FF00000000000000FF
      0000000000000000000000FF0000000000000000000000000000808080000000
      0000000000000000000000000000000000008080800000000000000000000000
      0000000000000000000000FF0000000000000000000000FF0000000000000000
      FF00000000000000FF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000FFFF00000000000000000000FFFF000000000000FFFF
      00000000000000000000FFFF000000000000FFFF0000FFFF0000000000000000
      0000000000008080800000000000000000000000000000000000000000000000
      000000000000808080000000000000000000000000000000000000FF00000000
      00000000000000FF0000000000000000FF00000000000000FF00000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000FFFF00000000
      000000000000FFFF000000000000FFFF00000000000000000000FFFF00000000
      0000FFFF0000FFFF0000000000000000000000000000808080000000000000FF
      FF000000000000FFFF000000000000FFFF000000000080808000000000000000
      0000000000000000000000FF0000000000000000000000FF0000000000000000
      FF00000000000000FF0000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000FFFF00000000000000000000FFFF000000000000FFFF
      00000000000000000000FFFF000000000000FFFF0000FFFF0000000000000000
      0000000000008080800000000000000000000000000000000000000000000000
      000000000000808080000000000000000000000000000000000000FF00000000
      000000FF00000000000000000000000000000000FF0000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000FFFF00000000
      000000000000FFFF000000000000FFFF00000000000000000000FFFF00000000
      0000FFFF0000FFFF0000000000000000000000000000808080000000000000FF
      00000000000000FF00000000000000FFFF000000000080808000000000000000
      0000000000000000000000FF00000000000000FF000000000000000000000000
      00000000FF000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000FF0000000000000000000000
      0000000000000000FF0000000000000000000000FF00000000000000FF000000
      0000000000008080800000000000000000000000000000000000000000000000
      000000000000808080000000000000000000000000000000000000FF00000000
      000000FF00000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000FF00000000000000000000000000000000000000FF00000000000000
      00000000FF00000000000000FF000000000000000000808080000000000000FF
      00000000000000FF00000000000000FFFF000000000080808000000000000000
      000000000000000000000000000000FF00000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000FF0000000000000000000000
      0000000000000000FF0000000000000000000000FF00000000000000FF000000
      0000000000008080800000000000000000000000000000000000000000000000
      00000000000080808000000000000000000000000000000000000000000000FF
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000FF00000000000000000000000000000000000000FF00000000000000
      00000000FF00000000000000FF000000000000000000808080000000000000FF
      FF000000000000FFFF000000000000FFFF000000000080808000000000000000
      0000424D3E000000000000003E00000028000000340000002700000001000100
      00000000380100000000000000000000000000000000000000000000FFFFFF00
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000FFDFE23FFFE0F000FEDFF775ADE0F000
      FD2FF775ADC0700001202FB5ADC070003981EFB5ADC07000DABCE75AD3803000
      DABE705AD3803000DABE7FDAD3803000D77CFADAD3803000D779FDF7B5803000
      D7F83AF7B5803000EFFFFFF7B5803000EFFFF8F7B58030000000000000000000
      0000000000000000000000000000}
  end
  object ExportDataDialog: TSaveDialog
    InitialDir = 'c:\desktop'
    Title = 'Export surf data to file...'
    Left = 390
    Top = 30
  end
  object PlayTimer: TTimer
    Enabled = False
    Interval = 100
    OnTimer = PlayTimerTimer
    Left = 376
    Top = 30
  end
end
