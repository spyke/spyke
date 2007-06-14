{==============================================================================|
| Project : Delphree - AsyncFree                                 | 001.003.000 |
|==============================================================================|
| Content:  Registering components property editors                            |
|==============================================================================|
| The contents of this file are subject to the Mozilla Public License Ver. 1.0 |
| (the "License"); you may not use this file except in compliance with the     |
| License. You may obtain a copy of the License at http://www.mozilla.org/MPL/ |
|                                                                              |
| Software distributed under the License is distributed on an "AS IS" basis,   |
| WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License for |
| the specific language governing rights and limitations under the License.    |
|==============================================================================|
| The Original Code is AsyncFree Library.                                      |
|==============================================================================|
| The Initial Developer of the Original Code is Petr Vones (Czech Republic).   |
| Portions created by Petr Vones are Copyright (C) 1998, 1999.                 |
| All Rights Reserved.                                                         |
|==============================================================================|
| Contributor(s):                                                              |
|==============================================================================|
| History:                                                                     |
|  7.03.1999 Version 001.000.000                                               |
|            Beta Preview, still development version.                          |
| 14.04.1999 Version 001.001.000                                               |
|            TAfDataTermial component added                                    |
| 29.06.1999 Version 001.002.000                                               |
|            AfTerminal LF bug fixed                                           |
|            Some changes in TAfLineViewer painting and font metrics           |
|            TAfLineViewer horizontal scrolling flickers fixed                 |
|            TAfTerminal - BkSpcMode and SelectedStyle properties added        |
|            TAfFileViewer component added                                     |
| 16.10.1999 Version 001.002.00                                                |
|            TAfCustomComPort.ExecuteConfigDialog bug fixed                    |
|            Delphi 5 compatibility checked                                    |
|==============================================================================}

unit AfRegister;

interface

{$I PVDEFINE.INC}

uses
  Classes, AfComPort, AfDataDispatcher, AfDataTerminal, AfPortControls;

procedure Register;

implementation

uses DsgnIntf;

{$R AfRegister.dcr}

type
  TAfNosortEnumProperty = class(TEnumProperty)
  public
    function GetAttributes: TPropertyAttributes; override;
  end;

function TAfNosortEnumProperty.GetAttributes: TPropertyAttributes;
begin
  Result := inherited GetAttributes - [paSortList];
end;

procedure Register;
begin
  RegisterComponents('AsyncFree', [TAfComPort, TAfDataDispatcher, TAfDataTerminal,
    TAfPortComboBox, TAfPortRadioGroup]);
  RegisterPropertyEditor(TypeInfo(TAfBaudrate), TAfCustomComPort, 'BaudRate', TAfNosortEnumProperty);
  RegisterPropertyEditor(TypeInfo(TThreadPriority), TAfCustomComPort, 'EventThreadPriority', TAfNosortEnumProperty);
  RegisterPropertyEditor(TypeInfo(TAfFlowControl), TAfCustomComPort, 'FlowControl', TAfNosortEnumProperty);
  RegisterPropertyEditor(TypeInfo(TAfParity), TAfCustomComPort, 'Parity', TAfNosortEnumProperty);
  RegisterPropertyEditor(TypeInfo(TThreadPriority), TAfCustomComPort, 'WriteThreadPriority', TAfNosortEnumProperty);
end;

end.
