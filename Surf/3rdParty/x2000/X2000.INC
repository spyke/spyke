{ ****************************************************************
  Source File Name   :  X2000.INC
  Decription         :  Const variables for X2000

  Please don´t modified this file. X2000 need this one.
**************************************************************** }
Type
  XTab = array[1..6] of string;

const
  cSystemRes = 0;
  cGDIRes = 1;
  cUSERRes = 2;

  PrgV_SI  = '1.13';
  PrgV_DI  = '1.03';
  PrgV_RC  = '1.01';
  PrgV_FC  = '1.01';
  PrgV_SP  = '1.0';
  PrgV_LA  = '1.04';
  
  ProgrammBuild   = '09.09.1999';
  ProgrammVersion = '1.02a';

  MaxPrograms     = 6;
  XPrograms: XTab = ('TSystemInfo2000X, Version '+PrgV_SI,'TDiskInfo2000X, Version '+PrgV_DI,'TRemoveCaption2000X, Version '+PrgV_RC,'TFileCompress2000X, Version '+PrgV_FC,'TScrPass2000X, Version '+PrgV_SP, 'TLabel2000X, Version '+PrgV_LA);

  Root1=HKEY_LOCAL_MACHINE;
  Root2=HKEY_CURRENT_USER;

  //Win98 Keys
  DK1                    ='System\CurrentControlSet\Services\';
  DK2                    ='SOFTWARE\Microsoft\';
  DK3                    =DK2+'Windows\CurrentVersion\';

  WIN98_CPU_Key          ='Hardware\Description\System\CentralProcessor\0';
  WIN98_Display_Key1     ='Config\0001\Display\Settings';
  WIN98_Display_Key2     =DK1+'Class\Display\0000';
  WIN98_Bios_Key         ='Enum\Root\*PNP0C01\0000';
  WIN98_Name_Key         =DK1+'VxD\VNETSUP';
  WIN98_Win_Key          =DK2+'Windows\CurrentVersion';
  WIN98_DirectX_KEY      =DK2+'DirectX';
  WIN98_Monitor_Key      =DK1+'Class\Monitor\0000';
  WIN98_Mouse_Key1       =DK1+'Class\Mouse\0000';
  WIN98_Mouse_Key2       ='Control Panel\Mouse';
  WIN98_Keyboard_Key     =DK1+'Class\Keyboard\0000';
  WIN98_Media_Key1       =DK1+'Class\Media';
  WIN98_Media_Key2       ='System\CurrentControlSet\Control\MediaResources';
  WIN98_Modem_Key        =DK1+'Class\Modem';
  WIN98_Path_Key         =DK3+'Explorer\Shell Folders';
  WIN98_CoProcessor_Key  ='Enum\Bios\*PNP0C04\0B';
  WIN98_IDE_KEY1         =DK1+'Class\hdc\0000';
  WIN98_IDE_KEY2         =DK1+'Class\hdc\0001';
  WIN98_USB_Key          =DK1+'Class\USB\0000';
  WIN98_SCSI_Key         =DK1+'Class\SCSIAdapter\0000';
  WIN98_System_Key       =DK1+'Class\System';
  WIN98_Net_Key1         =DK1+'Class\Net';
  WIN98_Net_Key2         =DK1+'Class\NetClient';
  WIN98_Net_Key3         =DK1+'Class\NetService';
  WIN98_NetProtocol_Key  =DK1+'Class\NetTrans';
  WIN98_Email_Key        =DK2+'Internet Account Manager\Accounts';
  WIN98_Proxy_Key        =DK3+'Internet Settings';
  WIN98_UnInstall_Key    =DK3+'UnInstall';
  WIN98_Fonts_Key        =DK3+'Fonts';
  WIN98_Direct3D_Key     =DK2+'Direct3D\Drivers';
  WIN98_DirectMusic_Key  =DK2+'DirectMusic';
  WIN98_DirectPlay_Key   =DK2+'DirectPlay\Service Providers';


  //NT Keys 
  WINNT_Win_Key          =DK2+'Windows NT\CurrentVersion';
  WINNT_Bios_Key         ='HARDWARE\DESCRIPTION\System';


