object TemplateWin: TTemplateWin
  Left = 1954
  Top = 96
  BorderStyle = bsSingle
  Caption = 'Templates'
  ClientHeight = 295
  ClientWidth = 292
  Color = clBtnFace
  Constraints.MinHeight = 250
  Constraints.MinWidth = 300
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -13
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  KeyPreview = True
  OldCreateOrder = False
  OnCreate = FormCreate
  OnKeyDown = FormKeyDown
  OnKeyUp = FormKeyUp
  PixelsPerInch = 96
  TextHeight = 16
  object TabControl: TTabControl
    Left = 0
    Top = 0
    Width = 292
    Height = 295
    Align = alClient
    Constraints.MinHeight = 295
    Constraints.MinWidth = 260
    HotTrack = True
    MultiSelect = True
    TabOrder = 0
    TabPosition = tpBottom
    Tabs.Strings = (
      'Cfg')
    TabIndex = 0
    TabWidth = 25
    OnChange = TabControlChange
    OnMouseDown = TabControlMouseDown
    object TabImage: TImage
      Left = 4
      Top = 48
      Width = 284
      Height = 222
      Align = alClient
    end
    object TabPanel: TPanel
      Left = 4
      Top = 6
      Width = 284
      Height = 42
      Align = alTop
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clWindowText
      Font.Height = -13
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
      TabOrder = 0
      Visible = False
      object Label3: TLabel
        Left = 3
        Top = 2
        Width = 24
        Height = 13
        Caption = 'n = 0'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clBlue
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentFont = False
      end
      object Label1: TLabel
        Left = 174
        Top = 2
        Width = 31
        Height = 13
        Caption = '50 rms'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentFont = False
      end
      object Label2: TLabel
        Left = 3
        Top = 21
        Width = 63
        Height = 13
        Alignment = taRightJustify
        Caption = 'Fit threshold: '
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clBlue
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentFont = False
      end
      object cbLocked: TCheckBox
        Left = 227
        Top = 3
        Width = 45
        Height = 14
        Caption = 'Lock'
        Color = clBtnFace
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clBlue
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentColor = False
        ParentFont = False
        TabOrder = 0
        OnClick = cbLockedClick
      end
      object cbEnabled: TCheckBox
        Left = 227
        Top = 19
        Width = 60
        Height = 14
        Caption = 'Enabled'
        Checked = True
        Color = clBtnFace
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clBlue
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentColor = False
        ParentFont = False
        State = cbChecked
        TabOrder = 1
        OnClick = cbEnabledClick
      end
      object tbDump: TButton
        Left = 80
        Top = 2
        Width = 28
        Height = 15
        Caption = 'Dump'
        Font.Charset = ANSI_CHARSET
        Font.Color = clWindowText
        Font.Height = -9
        Font.Name = 'Small Fonts'
        Font.Style = []
        ParentFont = False
        TabOrder = 2
        OnClick = tbDumpClick
      end
      object tbRawAvg: TButton
        Left = 109
        Top = 2
        Width = 40
        Height = 15
        Caption = 'Raw'
        Font.Charset = ANSI_CHARSET
        Font.Color = clGray
        Font.Height = -9
        Font.Name = 'Small Fonts'
        Font.Style = []
        ParentFont = False
        TabOrder = 3
        OnClick = tbRawAvgClick
      end
      object UpDown1: TUpDown
        Left = 149
        Top = 1
        Width = 20
        Height = 16
        Min = 0
        Max = 32767
        Orientation = udHorizontal
        Position = 0
        TabOrder = 4
        Wrap = False
        OnClick = UpDown1Click
      end
      object tbDel: TButton
        Left = 49
        Top = 2
        Width = 25
        Height = 15
        Caption = 'Del'
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -9
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentFont = False
        TabOrder = 5
        OnClick = tbDelClick
      end
      object udNoise: TUpDown
        Left = 208
        Top = 1
        Width = 16
        Height = 16
        Min = 0
        Max = 1000
        Increment = 5
        Position = 50
        TabOrder = 6
        Wrap = False
        OnClick = udNoiseClick
      end
      object seFitThresh: TSpinEdit
        Left = 69
        Top = 19
        Width = 67
        Height = 19
        AutoSize = False
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clWindowText
        Font.Height = -9
        Font.Name = 'Small Fonts'
        Font.Style = []
        Increment = 10000
        MaxValue = 0
        MinValue = 0
        ParentFont = False
        TabOrder = 7
        Value = 50000
        OnChange = seFitThreshChange
      end
      object cbShowFits: TCheckBox
        Left = 147
        Top = 19
        Width = 65
        Height = 17
        Caption = 'Show fits'
        Color = clActiveBorder
        Enabled = False
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clBlue
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ParentColor = False
        ParentFont = False
        TabOrder = 8
      end
    end
    object AllPanel: TPanel
      Left = 4
      Top = 48
      Width = 284
      Height = 222
      Align = alClient
      BevelOuter = bvNone
      Font.Charset = DEFAULT_CHARSET
      Font.Color = clBlue
      Font.Height = -11
      Font.Name = 'MS Sans Serif'
      Font.Style = []
      ParentFont = False
      TabOrder = 1
      object rgFitMethod: TRadioGroup
        Left = 0
        Top = 0
        Width = 81
        Height = 54
        Caption = 'Fitting method '
        Color = clBtnFace
        Font.Charset = DEFAULT_CHARSET
        Font.Color = clBlue
        Font.Height = -11
        Font.Name = 'MS Sans Serif'
        Font.Style = []
        ItemIndex = 0
        Items.Strings = (
          'SSqrs.'
          'X-Cor.')
        ParentColor = False
        ParentFont = False
        TabOrder = 0
      end
      object rgErWeight: TRadioGroup
        Left = 84
        Top = 1
        Width = 80
        Height = 54
        Caption = 'Error weighting '
        Font.Charset = ANSI_CHARSET
        Font.Color = clBlue
        Font.Height = -9
        Font.Name = 'Small Fonts'
        Font.Style = []
        ItemIndex = 0
        Items.Strings = (
          'Unweighted'
          'StdDev. '
          'RMS')
        ParentFont = False
        TabOrder = 1
      end
      object Panel1: TPanel
        Left = 0
        Top = 57
        Width = 281
        Height = 200
        Hint = 'Auto-shrink templates to size'
        BevelInner = bvRaised
        BevelOuter = bvLowered
        ParentShowHint = False
        ShowHint = True
        TabOrder = 2
        object lblNTemplates: TLabel
          Left = 2
          Top = 104
          Width = 100
          Height = 13
          Caption = 'Number of templates:'
        end
        object lbNumTemplates: TLabel
          Left = 105
          Top = 105
          Width = 6
          Height = 13
          Caption = '0'
        end
        object Label5: TLabel
          Left = 55
          Top = 66
          Width = 71
          Height = 13
          Hint = '(µm)'
          Caption = 'template radius'
        end
        object Label6: TLabel
          Left = 55
          Top = 42
          Width = 75
          Height = 13
          Caption = 'spikes/template'
        end
        object Label7: TLabel
          Left = 2
          Top = 137
          Width = 140
          Height = 13
          Caption = 'Time since new template (s) : '
        end
        object Label9: TLabel
          Left = 102
          Top = 84
          Width = 61
          Height = 13
          Caption = 'duration (s) : '
        end
        object lbSampleTime: TLabel
          Left = 164
          Top = 85
          Width = 6
          Height = 13
          Caption = '0'
        end
        object Label10: TLabel
          Left = 141
          Top = 138
          Width = 6
          Height = 13
          Caption = '0'
        end
        object Label4: TLabel
          Left = 152
          Top = 64
          Width = 46
          Height = 13
          Caption = 'View by...'
        end
        object Label11: TLabel
          Left = 195
          Top = 38
          Width = 73
          Height = 13
          Caption = '...fistogram bins'
        end
        object Label12: TLabel
          Left = 2
          Top = 120
          Width = 102
          Height = 13
          Caption = 'Number of samples  : '
        end
        object lbNumSamples: TLabel
          Left = 105
          Top = 121
          Width = 6
          Height = 13
          Caption = '0'
        end
        object Label8: TLabel
          Left = 9
          Top = 179
          Width = 110
          Height = 13
          Caption = 'nsamples   max classes'
        end
        object seTempRadius: TSpinEdit
          Left = 3
          Top = 61
          Width = 49
          Height = 22
          Hint = 'Radius of channels to include in template'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clBlack
          Font.Height = -12
          Font.Name = 'MS Sans Serif'
          Font.Style = []
          Increment = 5
          MaxValue = 2000
          MinValue = 50
          ParentFont = False
          ParentShowHint = False
          ShowHint = True
          TabOrder = 0
          Value = 75
          OnChange = seTempRadiusChange
        end
        object seMaxSpikes: TSpinEdit
          Left = 3
          Top = 37
          Width = 49
          Height = 22
          Hint = 'Number of spike events before template is automatically locked'
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clBlack
          Font.Height = -12
          Font.Name = 'MS Sans Serif'
          Font.Style = []
          Increment = 5
          MaxValue = 1000
          MinValue = 1
          ParentFont = False
          ParentShowHint = False
          ShowHint = True
          TabOrder = 1
          Value = 500
        end
        object cbRandomSample: TCheckBox
          Left = 3
          Top = 83
          Width = 99
          Height = 17
          Hint = 'Check to build templates from random sample of whole file'
          Caption = 'random sample...'
          Checked = True
          ParentShowHint = False
          ShowHint = True
          State = cbChecked
          TabOrder = 2
        end
        object cbViewOrder: TComboBox
          Left = 201
          Top = 56
          Width = 60
          Height = 22
          Style = csDropDownList
          Font.Charset = ANSI_CHARSET
          Font.Color = clMenuText
          Font.Height = -11
          Font.Name = 'Arial'
          Font.Style = []
          ItemHeight = 14
          ParentFont = False
          TabOrder = 3
          OnChange = cbViewOrderChange
          Items.Strings = (
            'maxch'
            'active'
            'dec n'
            'inc n'
            'dist')
        end
        object tbExtractRaw: TButton
          Left = 160
          Top = 106
          Width = 113
          Height = 25
          Caption = 'Restore  Waveforms'
          Enabled = False
          TabOrder = 4
          OnClick = tbExtractRawClick
        end
        object tbBuildTemplates: TButton
          Left = 7
          Top = 5
          Width = 268
          Height = 29
          Caption = 'Build Templates'
          ParentShowHint = False
          ShowHint = False
          TabOrder = 5
          OnClick = tbBuildTemplatesClick
        end
        object tbShrinkAll: TButton
          Left = 198
          Top = 88
          Width = 75
          Height = 14
          Caption = 'Shrink All'
          TabOrder = 6
          OnClick = tbShrinkAllClick
        end
        object seNHistBins: TSpinEdit
          Left = 144
          Top = 36
          Width = 49
          Height = 22
          Font.Charset = DEFAULT_CHARSET
          Font.Color = clBlack
          Font.Height = -11
          Font.Name = 'MS Sans Serif'
          Font.Style = []
          Increment = 50
          MaxValue = 1000
          MinValue = 0
          ParentFont = False
          TabOrder = 7
          Value = 250
        end
        object seNSamp: TSpinEdit
          Left = 6
          Top = 158
          Width = 51
          Height = 19
          Hint = 'Numbe of spike samples in training set'
          Font.Charset = ANSI_CHARSET
          Font.Color = clBlack
          Font.Height = -9
          Font.Name = 'Small Fonts'
          Font.Style = []
          Increment = 50
          MaxValue = 20000
          MinValue = 1
          ParentFont = False
          ParentShowHint = False
          ShowHint = True
          TabOrder = 8
          Value = 500
        end
        object seMaxClust: TSpinEdit
          Left = 69
          Top = 158
          Width = 43
          Height = 19
          Hint = 'Number of starting classes, k'
          Font.Charset = ANSI_CHARSET
          Font.Color = clBlack
          Font.Height = -9
          Font.Name = 'Small Fonts'
          Font.Style = []
          Increment = 2
          MaxValue = 200
          MinValue = 1
          ParentFont = False
          ParentShowHint = False
          ShowHint = True
          TabOrder = 9
          Value = 20
        end
        object cbGlobalEnable: TCheckBox
          Left = 160
          Top = 140
          Width = 113
          Height = 17
          Caption = 'global (dis/en)-able'
          Checked = True
          State = cbChecked
          TabOrder = 10
          OnClick = cbGlobalEnableClick
        end
      end
      object tbTemplate: TToolBar
        Left = 192
        Top = 224
        Width = 80
        Height = 24
        Align = alNone
        AutoSize = True
        ButtonWidth = 25
        Caption = 'tbTemplate'
        EdgeBorders = []
        Images = tbImages
        ParentShowHint = False
        ShowHint = False
        TabOrder = 3
        object tbOpenFile: TToolButton
          Left = 0
          Top = 2
          Hint = 'Open template file'
          Caption = 'tbOpenFile'
          ImageIndex = 0
          ParentShowHint = False
          ShowHint = True
          OnClick = tbOpenFileClick
        end
        object tbSaveFile: TToolButton
          Left = 25
          Top = 2
          Hint = 'Save current templates'
          Caption = 'tbSaveFile'
          ImageIndex = 1
          ParentShowHint = False
          ShowHint = True
          OnClick = tbSaveFileClick
        end
        object ToolButton1: TToolButton
          Left = 50
          Top = 2
          Width = 5
          Caption = 'ToolButton1'
          ImageIndex = 3
          Style = tbsSeparator
        end
        object tbReset: TToolButton
          Left = 55
          Top = 2
          Hint = 'Clear all templates'
          Caption = 'tbReset'
          ImageIndex = 2
          ParentShowHint = False
          ShowHint = True
          OnClick = tbResetClick
        end
      end
      object rgChartDisp: TRadioGroup
        Left = 167
        Top = 0
        Width = 114
        Height = 55
        Caption = 'Chart display '
        ItemIndex = 1
        Items.Strings = (
          'None'
          'Show Templates'
          'Hide Templates')
        TabOrder = 4
      end
    end
  end
  object tbImages: TImageList
    Left = 193
    Top = 342
    Bitmap = {
      494C010103000400040010001000FFFFFFFFFF10FFFFFFFFFFFFFFFF424D3600
      0000000000003600000028000000400000001000000001002000000000000010
      000000000000000000000000000000000000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000C0C0C000C0C0C00000808000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000000000000000000000000000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C00000000000008080000080
      8000000000000000000000000000000000000000000000000000C0C0C000C0C0
      C000000000000080800000000000C0C0C0000080800000FFFF0000FFFF0000FF
      FF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FF
      FF0000FFFF0000FFFF0000000000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000008080000080
      8000008080000080800000808000008080000080800000808000008080000000
      0000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C00000000000008080000080
      8000000000000000000000000000000000000000000000000000C0C0C000C0C0
      C000000000000080800000000000C0C0C0000080800000FFFF0000FFFF0000FF
      FF0000FFFF0000FFFF0080808000000000008080800000FFFF0000FFFF0000FF
      FF0000FFFF0000FFFF0000000000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000FFFF00000000000080
      8000008080000080800000808000008080000080800000808000008080000080
      800000000000C0C0C000C0C0C000C0C0C000C0C0C00000000000008080000080
      8000000000000000000000000000000000000000000000000000C0C0C000C0C0
      C000000000000080800000000000C0C0C0000080800000FFFF0000FFFF0000FF
      FF0000FFFF0000FFFF0080808000000000008080800000FFFF0000FFFF0000FF
      FF0000FFFF0000FFFF0000000000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FFFFFF0000FFFF000000
      0000008080000080800000808000008080000080800000808000008080000080
      80000080800000000000C0C0C000C0C0C000C0C0C00000000000008080000080
      8000000000000000000000000000000000000000000000000000000000000000
      0000000000000080800000000000C0C0C000C0C0C0000080800000FFFF0000FF
      FF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FF
      FF0000FFFF0000000000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000FFFF00FFFFFF0000FF
      FF00000000000080800000808000008080000080800000808000008080000080
      8000008080000080800000000000C0C0C000C0C0C00000000000008080000080
      8000008080000080800000808000008080000080800000808000008080000080
      8000008080000080800000000000C0C0C000C0C0C0000080800000FFFF0000FF
      FF0000FFFF0000FFFF0000FFFF000000000000FFFF0000FFFF0000FFFF0000FF
      FF0000FFFF0000000000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FFFFFF0000FFFF00FFFF
      FF0000FFFF000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000C0C0C00000000000008080000080
      8000000000000000000000000000000000000000000000000000000000000000
      0000008080000080800000000000C0C0C000C0C0C000C0C0C0000080800000FF
      FF0000FFFF0000FFFF0000808000000000000080800000FFFF0000FFFF0000FF
      FF0000000000C0C0C000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000FFFF00FFFFFF0000FF
      FF00FFFFFF0000FFFF00FFFFFF0000FFFF00FFFFFF0000FFFF0000000000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C00000000000008080000000
      0000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000000000000080800000000000C0C0C000C0C0C000C0C0C0000080800000FF
      FF0000FFFF0000FFFF0000000000000000000000000000FFFF0000FFFF0000FF
      FF0000000000C0C0C000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      00000000000000000000000000000000000000000000FFFFFF0000FFFF00FFFF
      FF0000FFFF00FFFFFF0000FFFF00FFFFFF0000FFFF00FFFFFF0000000000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C00000000000008080000000
      0000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000000000000080800000000000C0C0C000C0C0C000C0C0C000C0C0C0000080
      800000FFFF0000FFFF0000000000000000000000000000FFFF0000FFFF000000
      0000C0C0C000C0C0C000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000FFFF00FFFFFF0000FF
      FF0000000000000000000000000000000000000000000000000000000000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C00000000000008080000000
      0000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000000000000080800000000000C0C0C000C0C0C000C0C0C000C0C0C0000080
      800000FFFF0000FFFF0000000000000000000000000000FFFF0000FFFF000000
      0000C0C0C000C0C0C000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000C0C0C00000000000000000000000
      0000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000000000000000000000000000C0C0C000C0C0C00000000000008080000000
      0000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000000000000080800000000000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C0000080800000FFFF0000FFFF000000000000FFFF0000FFFF0000000000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C0000000000000000000C0C0C000C0C0C00000000000008080000000
      0000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000000000000000000000000000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C0000080800000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000000000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C00000000000C0C0C000C0C0C000C0C0
      C00000000000C0C0C00000000000C0C0C000C0C0C00000000000008080000000
      0000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C00000000000C0C0C00000000000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C0000080800000FFFF0000FFFF0000FFFF0000000000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C00000000000000000000000
      0000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C00000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C0000080800000FFFF0000FFFF0000FFFF0000000000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000008080000080800000808000C0C0C000C0C0C000C0C0
      C000C0C0C000C0C0C000C0C0C000C0C0C0000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000000000000000000000000000424D3E000000000000003E000000
      2800000040000000100000000100010000000000800000000000000000000000
      000000000000000000000000FFFFFF0000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      0000000000000000000000000000000000000000000000000000000000000000
      000000000000}
  end
  object SaveTemplates: TSaveDialog
    Left = 222
    Top = 342
  end
  object OpenTemplates: TOpenDialog
    Left = 252
    Top = 342
  end
end
