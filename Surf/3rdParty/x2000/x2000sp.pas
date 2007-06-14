{ ****************************************************************
  Info               :  TScrPass2000X
                        Freeware

  Source File Name   :  X2000SP.PAS
  Author             :  Baldemaier Florian(Baldemaier.Florian@gmx.net)
  Original made by   :  Andreas Windisch (wist@gmx.de)  AW
  Compiler           :  Delphi 4.0 Client/Server, Service Pack 3
  Decription         :  It show´s you the password of the current
                        Screensaver.

  Testet on          :  Intel Pentium II, 300 Mhz and 450 Mhz, Windows 98
                        Intel Pentium III, 500 Mhz, Windows 98
                        Intel Pentium 200 Mhz, Windows98
                        D4 Service Pack 3
                        Mircosoft Windows 98
                        Mircosoft Windows 98 SE

 Important !!!!!!!!!!!!!
 Use this program at your own discretion. The author cannot be held
 responsible for any loss of data, or other misfortunes resulting 
 from the use of this program that may or may not occur. Likewise, the     
 author cannot be held responsible for the use, or misuse of this
 software. I have no clue on how well it will work or even IF it
 will work for that matter. 
**************************************************************** }
unit x2000sp;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs, Registry, DsgnIntf;

type
  TTabelle = array[0..11] of string;

  TAbout2000X=class(TPropertyEditor)
  public
   procedure Edit; override;
   function GetAttributes: TPropertyAttributes; override;
   function GetValue: string; override;
  end;

  TScrPass2000X = class(TComponent)
  private
    FAbout: TAbout2000X;
    function Buchstabe(Zahlen : String; Reihe : Integer) : String;
  public
    constructor create(AOwner:TComponent);
    function GetPassword : String;
  published
    property About          : TAbout2000X  read FAbout  write FAbout;
  end;


const
  Count: Integer = 0;
  Tabelle_A   : TTabelle = ('3039','4146','3337','3543','3236','3238','4530','3541','3342','4344','3036','4239');
  Tabelle_B   : TTabelle = ('3041','4143','3334','3546','3235','3242','4533','3539','3338','4345','3035','4241');
  Tabelle_C   : TTabelle = ('3042','4144','3335','3545','3234','3241','4532','3538','3339','4346','3034','4242');
  Tabelle_D   : TTabelle = ('3043','4141','3332','3539','3233','3244','4535','3546','3345','4338','3033','4243');
  Tabelle_E   : TTabelle = ('3044','4142','3333','3538','3232','3243','4534','3545','3346','4339','3032','4244');
  Tabelle_F   : TTabelle = ('3045','4138','3330','3542','3231','3246','4537','3544','3343','4341','3031','4245');
  Tabelle_G   : TTabelle = ('3046','4139','3331','3541','3230','3245','4536','3543','3344','4342','3030','4246');
  Tabelle_H   : TTabelle = ('3030','4136','3345','3535','3246','3231','4539','3533','3332','4334','3046','4230');
  Tabelle_I   : TTabelle = ('3031','4137','3346','3534','3245','3230','4538','3532','3333','4335','3045','4231');
  Tabelle_J   : TTabelle = ('3032','4134','3343','3537','3244','3233','4542','3531','3330','4336','3044','4232');
  Tabelle_K   : TTabelle = ('3033','4135','3344','3536','3243','3232','4541','3530','3331','4337','3043','4233');
  Tabelle_L   : TTabelle = ('3034','4132','3341','3531','3242','3235','4544','3537','3336','4330','3042','4234');
  Tabelle_M   : TTabelle = ('3035','4133','3342','3530','3241','3234','4543','3536','3337','4331','3041','4235');
  Tabelle_N   : TTabelle = ('3036','4130','3338','3533','3239','3237','4546','3535','3334','4332','3039','4236');
  Tabelle_O   : TTabelle = ('3037','4131','3339','3532','3238','3236','4545','3534','3335','4333','3038','4237');
  Tabelle_P   : TTabelle = ('3138','4245','3236','3444','3337','3339','4631','3442','3241','4443','3137','4138');
  Tabelle_Q   : TTabelle = ('3139','4246','3237','3443','3336','3338','4630','3441','3242','4444','3136','4139');
  Tabelle_R   : TTabelle = ('3141','4243','3234','3446','3335','3342','4633','3439','3238','4445','3135','4141');
  Tabelle_S   : TTabelle = ('3142','4244','3235','3445','3334','3341','4632','3438','3239','4446','3134','4142');
  Tabelle_T   : TTabelle = ('3143','4241','3232','3439','3333','3344','4635','3446','3245','4438','3133','4143');
  Tabelle_U   : TTabelle = ('3144','4242','3233','3438','3332','3343','4634','3445','3246','4439','3132','4144');
  Tabelle_V   : TTabelle = ('3145','4238','3230','3442','3331','3346','4637','3444','3243','4441','3131','4145');
  Tabelle_W   : TTabelle = ('3146','4239','3231','3441','3330','3345','4636','3443','3244','4442','3130','4146');
  Tabelle_X   : TTabelle = ('3130','4236','3245','3435','3346','3331','4639','3433','3232','4434','3146','4130');
  Tabelle_Y   : TTabelle = ('3131','4237','3246','3434','3345','3330','4638','3432','3233','4435','3145','4131');
  Tabelle_Z   : TTabelle = ('3132','4234','3243','3437','3344','3333','4642','3431','3230','4436','3144','4132');
  Tabelle_S1  : TTabelle = ('3038','4145','3336','3544','3237','3239','4531','3542','3341','4343','3037','4238'); //@
  Tabelle_S2  : TTabelle = ('3133','4235','3244','3436','3343','3332','4641','3430','3231','4437','3143','4133'); //[
  Tabelle_S3  : TTabelle = ('3134','4232','3241','3431','3342','3335','4644','3437','3236','4430','3142','4134'); //\
  Tabelle_S4  : TTabelle = ('3135','4233','3242','3430','3341','3334','4643','3436','3237','4431','3141','4135'); //]
  Tabelle_S5  : TTabelle = ('3136','4230','3238','3433','3339','3337','4646','3435','3234','4432','3139','4136'); //^
  Tabelle_S6  : TTabelle = ('3137','4231','3239','3432','3338','3336','4645','3434','3235','4433','3138','4137'); //_
  Tabelle_S7  : TTabelle = ('3238','3845','3136','3744','3037','3039','4331','3742','3141','4543','3237','3938'); //`
  Tabelle_S8  : TTabelle = ('3333','3935','3044','3636','3143','3132','4441','3630','3031','4637','3343','3833'); //{
  Tabelle_S9  : TTabelle = ('3334','3932','3041','3631','3142','3135','4444','3637','3036','4630','3342','3834'); //|
  Tabelle_S10 : TTabelle = ('3335','3933','3042','3630','3141','3134','4443','3636','3037','4631','3341','3835'); //}
  Tabelle_S11 : TTabelle = ('3336','3930','3038','3633','3139','3137','4446','3635','3034','4632','3339','3836'); //~
  Tabelle_S12 : TTabelle = ('3638','4345','3536','3344','3437','3439','3831','3342','3541','4143','3637','4438'); //Space
  Tabelle_S13 : TTabelle = ('3639','4346','3537','3343','3436','3438','3830','3341','3542','4144','3636','4439'); //!
  Tabelle_S14 : TTabelle = ('3641','4343','3534','3346','3435','3442','3833','3339','3538','4145','3635','4441'); //"
  Tabelle_S15 : TTabelle = ('3642','4344','3535','3345','3434','3441','3832','3338','3539','4146','3634','4442'); //#
  Tabelle_S16 : TTabelle = ('3643','4341','3532','3339','3433','3444','3835','3346','3545','4138','3633','4443'); //$
  Tabelle_S17 : TTabelle = ('3644','4342','3533','3338','3432','3443','3834','3345','3546','4139','3632','4444'); //%
  Tabelle_S18 : TTabelle = ('3645','4338','3530','3342','3431','3446','3837','3344','3543','4141','3631','4445'); //&
  Tabelle_S19 : TTabelle = ('3646','4339','3531','3341','3430','3445','3836','3343','3544','4142','3630','4446'); //'
  Tabelle_S20 : TTabelle = ('3630','4336','3545','3335','3446','3431','3839','3333','3532','4134','3646','4430'); //(
  Tabelle_S21 : TTabelle = ('3631','4337','3546','3334','3445','3430','3838','3332','3533','4135','3645','4431'); //)
  Tabelle_S22 : TTabelle = ('3632','4334','3543','3337','3444','3433','3842','3331','3530','4136','3644','4432'); //*
  Tabelle_S23 : TTabelle = ('3633','4335','3544','3336','3443','3432','3841','3330','3531','4137','3643','4433'); //+
  Tabelle_S24 : TTabelle = ('3634','4332','3541','3331','3442','3435','3844','3337','3536','4130','3642','4434'); //,
  Tabelle_S25 : TTabelle = ('3635','4333','3542','3330','3441','3434','3843','3336','3537','4131','3641','4435'); //-
  Tabelle_S26 : TTabelle = ('3636','4330','3538','3333','3439','3437','3846','3335','3534','4132','3639','4436'); //.
  Tabelle_S27 : TTabelle = ('3637','4331','3539','3332','3438','3436','3845','3334','3535','4133','3638','4437'); ///
  Tabelle_S28 : TTabelle = ('3732','4434','3443','3237','3544','3533','3942','3231','3430','4236','3744','4332'); //:
  Tabelle_S29 : TTabelle = ('3733','4435','3444','3236','3543','3532','3941','3230','3431','4237','3743','4333'); //;
  Tabelle_S30 : TTabelle = ('3734','4432','3441','3231','3542','3535','3944','3237','3436','4230','3742','4334'); //<
  Tabelle_S31 : TTabelle = ('3735','4433','3442','3230','3541','3534','3943','3236','3437','4231','3741','4335'); //=
  Tabelle_S32 : TTabelle = ('3736','4430','3438','3233','3539','3537','3946','3235','3434','4232','3739','4336'); //>
  Tabelle_S33 : TTabelle = ('3738','4445','3436','3244','3537','3539','3931','3242','3441','4243','3737','4338'); //0
  Tabelle_S34 : TTabelle = ('3739','4446','3437','3243','3536','3538','3930','3241','3442','4244','3736','4339'); //1
  Tabelle_S35 : TTabelle = ('3741','4443','3434','3246','3535','3542','3933','3239','3438','4245','3735','4341'); //2
  Tabelle_S36 : TTabelle = ('3742','4444','3435','3245','3534','3541','3937','3738','3439','4246','3734','4342'); //3
  Tabelle_S37 : TTabelle = ('3743','4441','3432','3239','3533','3544','3935','3246','3445','4238','3733','4343'); //4
  Tabelle_S38 : TTabelle = ('3744','4442','3433','3238','3532','3543','3934','3245','3446','4239','3732','4344'); //5
  Tabelle_S39 : TTabelle = ('3745','4438','3430','3242','3531','3546','3937','3244','3443','4241','3731','4345'); //6
  Tabelle_S40 : TTabelle = ('3746','4439','3431','3241','3530','3545','3936','3243','3444','4242','3730','4346'); //7
  Tabelle_S41 : TTabelle = ('3730','4436','3445','3235','3546','3531','3939','3233','3432','4234','3746','4330'); //8
  Tabelle_S42 : TTabelle = ('3731','4437','3446','3234','3545','3530','3938','3232','3433','4235','3745','4331'); //9
  Tabelle_S43 : TTabelle = ('3839','3246','4237','4443','4136','4138','3630','4441','4242','3444','3836','3339'); //á
  Tabelle_S44 : TTabelle = ('3843','3241','4232','4439','4133','4144','3635','4446','4245','3438','3833','3343'); //Ä
  Tabelle_S45 : TTabelle = ('3934','3332','4141','4331','4242','4235','3744','4337','4136','3530','3942','3234'); //Ü
  Tabelle_S46 : TTabelle = ('3935','3333','4142','4330','4241','4234','3743','4336','4137','3531','3941','3235'); //ý
  Tabelle_S47 : TTabelle = ('3945','3338','4130','4342','4231','4246','3737','4344','4143','3541','3931','3245'); //Ö
  Tabelle_S48 : TTabelle = ('4546','3439','4431','4241','4330','4345','3036','4243','4444','3242','4530','3546'); //§
  Tabelle_S49 : TTabelle = ('4638','3545','4336','4144','4437','4439','3131','4142','4341','3343','4637','3438'); //°
  Tabelle_S50 : TTabelle = ('4641','3543','4334','4146','4435','4442','3133','4139','4338','3345','4635','3441'); //²
  Tabelle_S51 : TTabelle = ('4642','3544','4335','4145','4434','4441','3132','4138','4339','3346','4634','3442'); //³
  Tabelle_S52 : TTabelle = ('4643','3541','4332','4139','4433','4444','3135','4146','4345','3338','4633','3443'); //´
  Tabelle_S53 : TTabelle = ('4644','3542','4333','4138','4432','4443','3134','4145','4346','3339','4632','3444'); //µ

implementation

uses X2000About;

function TScrPass2000X.Buchstabe(Zahlen : String; Reihe : Integer) : String;
begin
    if Tabelle_A[Reihe]   = Zahlen then Buchstabe := 'A';
    if Tabelle_B[Reihe]   = Zahlen then Buchstabe := 'B';
    if Tabelle_C[Reihe]   = Zahlen then Buchstabe := 'C';
    if Tabelle_D[Reihe]   = Zahlen then Buchstabe := 'D';
    if Tabelle_E[Reihe]   = Zahlen then Buchstabe := 'E';
    if Tabelle_F[Reihe]   = Zahlen then Buchstabe := 'F';
    if Tabelle_G[Reihe]   = Zahlen then Buchstabe := 'G';
    if Tabelle_H[Reihe]   = Zahlen then Buchstabe := 'H';
    if Tabelle_I[Reihe]   = Zahlen then Buchstabe := 'I';
    if Tabelle_J[Reihe]   = Zahlen then Buchstabe := 'J';
    if Tabelle_K[Reihe]   = Zahlen then Buchstabe := 'K';
    if Tabelle_L[Reihe]   = Zahlen then Buchstabe := 'L';
    if Tabelle_M[Reihe]   = Zahlen then Buchstabe := 'M';
    if Tabelle_N[Reihe]   = Zahlen then Buchstabe := 'N';
    if Tabelle_O[Reihe]   = Zahlen then Buchstabe := 'O';
    if Tabelle_P[Reihe]   = Zahlen then Buchstabe := 'P';
    if Tabelle_Q[Reihe]   = Zahlen then Buchstabe := 'Q';
    if Tabelle_R[Reihe]   = Zahlen then Buchstabe := 'R';
    if Tabelle_S[Reihe]   = Zahlen then Buchstabe := 'S';
    if Tabelle_T[Reihe]   = Zahlen then Buchstabe := 'T';
    if Tabelle_U[Reihe]   = Zahlen then Buchstabe := 'U';
    if Tabelle_V[Reihe]   = Zahlen then Buchstabe := 'V';
    if Tabelle_W[Reihe]   = Zahlen then Buchstabe := 'W';
    if Tabelle_X[Reihe]   = Zahlen then Buchstabe := 'X';
    if Tabelle_Y[Reihe]   = Zahlen then Buchstabe := 'Y';
    if Tabelle_Z[Reihe]   = Zahlen then Buchstabe := 'Z';
    if Tabelle_S1[Reihe]  = Zahlen then Buchstabe := '@';
    if Tabelle_S2[Reihe]  = Zahlen then Buchstabe := '[';
    if Tabelle_S3[Reihe]  = Zahlen then Buchstabe := '\';
    if Tabelle_S4[Reihe]  = Zahlen then Buchstabe := ']';
    if Tabelle_S5[Reihe]  = Zahlen then Buchstabe := '^';
    if Tabelle_S6[Reihe]  = Zahlen then Buchstabe := '_';
    if Tabelle_S7[Reihe]  = Zahlen then Buchstabe := '`';
    if Tabelle_S8[Reihe]  = Zahlen then Buchstabe := '{';
    if Tabelle_S9[Reihe]  = Zahlen then Buchstabe := '|';
    if Tabelle_S10[Reihe] = Zahlen then Buchstabe := '}';
    if Tabelle_S11[Reihe] = Zahlen then Buchstabe := '~';
    if Tabelle_S12[Reihe] = Zahlen then Buchstabe := ' ';
    if Tabelle_S13[Reihe] = Zahlen then Buchstabe := '!';
    if Tabelle_S14[Reihe] = Zahlen then Buchstabe := '"';
    if Tabelle_S15[Reihe] = Zahlen then Buchstabe := '#';
    if Tabelle_S16[Reihe] = Zahlen then Buchstabe := '$';
    if Tabelle_S17[Reihe] = Zahlen then Buchstabe := '%';
    if Tabelle_S18[Reihe] = Zahlen then Buchstabe := '&';
    if Tabelle_S19[Reihe] = Zahlen then Buchstabe := '''';
    if Tabelle_S20[Reihe] = Zahlen then Buchstabe := '(';
    if Tabelle_S21[Reihe] = Zahlen then Buchstabe := ')';
    if Tabelle_S22[Reihe] = Zahlen then Buchstabe := '*';
    if Tabelle_S23[Reihe] = Zahlen then Buchstabe := '+';
    if Tabelle_S24[Reihe] = Zahlen then Buchstabe := ',';
    if Tabelle_S25[Reihe] = Zahlen then Buchstabe := '-';
    if Tabelle_S26[Reihe] = Zahlen then Buchstabe := '.';
    if Tabelle_S27[Reihe] = Zahlen then Buchstabe := '/';
    if Tabelle_S28[Reihe] = Zahlen then Buchstabe := ':';
    if Tabelle_S29[Reihe] = Zahlen then Buchstabe := ';';
    if Tabelle_S30[Reihe] = Zahlen then Buchstabe := '<';
    if Tabelle_S31[Reihe] = Zahlen then Buchstabe := '=';
    if Tabelle_S32[Reihe] = Zahlen then Buchstabe := '>';
    if Tabelle_S33[Reihe] = Zahlen then Buchstabe := '0';
    if Tabelle_S34[Reihe] = Zahlen then Buchstabe := '1';
    if Tabelle_S35[Reihe] = Zahlen then Buchstabe := '2';
    if Tabelle_S36[Reihe] = Zahlen then Buchstabe := '3';
    if Tabelle_S37[Reihe] = Zahlen then Buchstabe := '4';
    if Tabelle_S38[Reihe] = Zahlen then Buchstabe := '5';
    if Tabelle_S39[Reihe] = Zahlen then Buchstabe := '6';
    if Tabelle_S40[Reihe] = Zahlen then Buchstabe := '7';
    if Tabelle_S41[Reihe] = Zahlen then Buchstabe := '8';
    if Tabelle_S42[Reihe] = Zahlen then Buchstabe := '9';
    if Tabelle_S43[Reihe] = Zahlen then Buchstabe := 'á';
    if Tabelle_S44[Reihe] = Zahlen then Buchstabe := 'Ä';
    if Tabelle_S45[Reihe] = Zahlen then Buchstabe := 'Ü';
    if Tabelle_S46[Reihe] = Zahlen then Buchstabe := 'ý';
    if Tabelle_S47[Reihe] = Zahlen then Buchstabe := 'Ö';
    if Tabelle_S48[Reihe] = Zahlen then Buchstabe := '§';
    if Tabelle_S49[Reihe] = Zahlen then Buchstabe := '°';
    if Tabelle_S50[Reihe] = Zahlen then Buchstabe := '²';
    if Tabelle_S51[Reihe] = Zahlen then Buchstabe := '³';
    if Tabelle_S52[Reihe] = Zahlen then Buchstabe := '´';
    if Tabelle_S53[Reihe] = Zahlen then Buchstabe := 'µ';
end;

constructor TScrPass2000X.create(AOwner:TComponent);
begin
  inherited create(AOwner);
//  Refresh;
end;

function TScrPass2000X.GetPassword : String;
var
  i, j : integer;
  Buffer: array[0..27] of Char;
  Tmp, Passwort, Daten, S : String;
  Lreg:TRegistry;
begin

  Lreg:=TRegistry.Create;
  with Lreg do
    try
       RootKey := HKEY_CURRENT_USER;

       If KeyExists('Control Panel\Desktop') then begin
         OpenKey('Control Panel\Desktop', false);

         ReadBinaryData('ScreenSave_Data', Buffer, Sizeof(Buffer));
         S := '';
         Daten := '';
         for i := 0 to 27 do S := S + (Buffer[i]);
         S := Trim(S);

         for i := 1 to 28 do begin
             Daten := Daten + IntToHex(Ord(S[i]),2);
             if copy(Daten,i*2-1,2) = '00' then break;
         end;

         for i := 1 to (Length(Daten) div 4) do begin
             j := 1; if i > 1 then j := j + ((i - 1) * 4) else j := 1;

             Tmp := Copy(Daten, j, 4);
             Passwort := Passwort + Buchstabe(Tmp, i-1);
         end;
         if (Passwort='') or (Passwort='0000000000') then Passwort:='(unavailable)';
         Result := Passwort;
       end;
    finally
       Free;
    end;
end;

procedure TAbout2000X.Edit;
begin
 with TAboutForm.Create(Application) do begin
  try
    ShowModal;
  finally
    Free;
  end;
 end;
end;

function TAbout2000X.GetAttributes: TPropertyAttributes;
begin
    Result := [paMultiSelect, paDialog, paReadOnly];
end;

function TAbout2000X.GetValue: string;
begin
    Result := '(X2000)';
end;

end.
