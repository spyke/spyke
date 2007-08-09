unit ElectrodeTypes;
interface
uses Windows;
const
  MAXCHANS = 64;
  MAXELECTRODEPOINTS = 12;
  KNOWNELECTRODES = 17;

type
  TElectrode = record
    NumPoints : integer;
    Outline : array[0..MAXELECTRODEPOINTS-1] of TPoint;  //in microns
    NumSites : Integer;
    SiteLoc : array[0..MAXCHANS-1] of TPoint; //in microns
    TopLeftSite,BotRightSite : TPoint;
    CenterX : Integer;
    SiteSize : TPoint; //in microns
    RoundSite,Created : boolean;
    Name : ShortString;
    Description : ShortString;
  end;

Function GetElectrode(var Electrode : TElectrode; Name : ShortString) : boolean;

var  KnownElectrode : array[0..KNOWNELECTRODES-1] of TElectrode;

implementation
Procedure MakeKnownElectrodes;
begin
    //Create the electrodes

    //PTRODE16a is PAH design
    With KnownElectrode[0] do
    begin
      Name := 'PTRODE16a';
      Description := '16 Chan Silicon, Staggered';
      SiteSize.x := 12;
      SiteSize.y := 12;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 10;
      Outline[0].x := -64;
      Outline[0].y := -50;
      Outline[1].x := -64;
      Outline[1].y := 465;
      Outline[2].x := -32;
      Outline[2].y := 527;
      Outline[3].x := -22;
      Outline[3].y := 589;
      Outline[4].x := 0;
      Outline[4].y := 639;
      Outline[5].x := 22;
      Outline[5].y := 589;
      Outline[6].x := 32;
      Outline[6].y := 527;
      Outline[7].x := 64;
      Outline[7].y := 465;
      Outline[8].x := 64;
      Outline[8].y := -50;
      Outline[9].x := Outline[0].x;
      Outline[9].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[0].x := -27;
      SiteLoc[0].y := 279;
      SiteLoc[1].x := -27;
      SiteLoc[1].y := 217;
      SiteLoc[2].x := -27;
      SiteLoc[2].y := 155;
      SiteLoc[3].x := -27;
      SiteLoc[3].y := 93;
      SiteLoc[4].x := -27;
      SiteLoc[4].y := 31;
      SiteLoc[5].x := -27;
      SiteLoc[5].y := 341;
      SiteLoc[6].x := -27;
      SiteLoc[6].y := 403;
      SiteLoc[7].x := -27;
      SiteLoc[7].y := 465;
      SiteLoc[8].x := 27;
      SiteLoc[8].y := 434;
      SiteLoc[9].x := 27;
      SiteLoc[9].y := 372;
      SiteLoc[10].x := 27;
      SiteLoc[10].y := 310;
      SiteLoc[11].x := 27;
      SiteLoc[11].y := 0;
      SiteLoc[12].x := 27;
      SiteLoc[12].y := 62;
      SiteLoc[13].x := 27;
      SiteLoc[13].y := 124;
      SiteLoc[14].x := 27;
      SiteLoc[14].y := 186;
      SiteLoc[15].x := 27;
      SiteLoc[15].y := 248;
    end;

    //PTRODE16b is PAH design, different channel layout
    With KnownElectrode[1] do
    begin
      Name := 'PTRODE16b';
      Description := '16 Chan Silicon, Staggered';
      SiteSize.x := 12;
      SiteSize.y := 12;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 10;
      Outline[0].x := -64;
      Outline[0].y := -50;
      Outline[1].x := -64;
      Outline[1].y := 465;
      Outline[2].x := -32;
      Outline[2].y := 527;
      Outline[3].x := -22;
      Outline[3].y := 589;
      Outline[4].x := 0;
      Outline[4].y := 639;
      Outline[5].x := 22;
      Outline[5].y := 589;
      Outline[6].x := 32;
      Outline[6].y := 527;
      Outline[7].x := 64;
      Outline[7].y := 465;
      Outline[8].x := 64;
      Outline[8].y := -50;
      Outline[9].x := Outline[0].x;
      Outline[9].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[0].x := -27;
      SiteLoc[0].y := 155;
      SiteLoc[1].x := -27;
      SiteLoc[1].y := 93;
      SiteLoc[2].x := -27;
      SiteLoc[2].y := 217;
      SiteLoc[3].x := -27;
      SiteLoc[3].y := 341;
      SiteLoc[4].x := -27;
      SiteLoc[4].y := 31;
      SiteLoc[5].x := -27;
      SiteLoc[5].y := 279;
      //6&7 have been swapped according to Jamie's instructions
      //(7&8, counting from 1)
      SiteLoc[6].x := -27;
      SiteLoc[6].y := 403;
      SiteLoc[7].x := -27;
      SiteLoc[7].y := 465;
      SiteLoc[8].x := 27;

      SiteLoc[8].y := 436;
      SiteLoc[9].x := 27;
      SiteLoc[9].y := 372;
      SiteLoc[10].x := 27;
      SiteLoc[10].y := 248;
      SiteLoc[11].x := 27;
      SiteLoc[11].y := 0;
      SiteLoc[12].x := 27;
      SiteLoc[12].y := 310;
      SiteLoc[13].x := 27;
      SiteLoc[13].y := 186;
      SiteLoc[14].x := 27;
      SiteLoc[14].y := 62;
      SiteLoc[15].x := 27;
      SiteLoc[15].y := 124;
    end;

    //PTRODE16x is 16a with left/right banks reversed
    With KnownElectrode[2] do
    begin
      Name := 'PTRODE16x';
      Description := '16 Chan Silicon, Staggered, Banks Crossed';
      SiteSize.x := 12;
      SiteSize.y := 12;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 10;
      Outline[0].x := -64;
      Outline[0].y := -50;
      Outline[1].x := -64;
      Outline[1].y := 465;
      Outline[2].x := -32;
      Outline[2].y := 527;
      Outline[3].x := -22;
      Outline[3].y := 589;
      Outline[4].x := 0;
      Outline[4].y := 639;
      Outline[5].x := 22;
      Outline[5].y := 589;
      Outline[6].x := 32;
      Outline[6].y := 527;
      Outline[7].x := 64;
      Outline[7].y := 465;
      Outline[8].x := 64;
      Outline[8].y := -50;
      Outline[9].x := Outline[0].x;
      Outline[9].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[8].x := -27;
      SiteLoc[8].y := 279;
      SiteLoc[9].x := -27;
      SiteLoc[9].y := 217;
      SiteLoc[10].x := -27;
      SiteLoc[10].y := 155;
      SiteLoc[11].x := -27;
      SiteLoc[11].y := 93;
      SiteLoc[12].x := -27;
      SiteLoc[12].y := 31;
      SiteLoc[13].x := -27;
      SiteLoc[13].y := 341;
      SiteLoc[14].x := -27;
      SiteLoc[14].y := 403;
      SiteLoc[15].x := -27;
      SiteLoc[15].y := 465;

      SiteLoc[0].x := 27;
      SiteLoc[0].y := 434;
      SiteLoc[1].x := 27;
      SiteLoc[1].y := 372;
      SiteLoc[2].x := 27;
      SiteLoc[2].y := 310;
      SiteLoc[3].x := 27;
      SiteLoc[3].y := 0;
      SiteLoc[4].x := 27;
      SiteLoc[4].y := 62;
      SiteLoc[5].x := 27;
      SiteLoc[5].y := 124;
      SiteLoc[6].x := 27;
      SiteLoc[6].y := 186;
      SiteLoc[7].x := 27;
      SiteLoc[7].y := 248;
    end;

    //PTRODE16c is PAH design, different channel mapping because of new HS27 headstage
    With KnownElectrode[3] do
    begin
      Name := 'PTRODE16c';
      Description := '16 Chan Staggered, 62µm spacing';
      SiteSize.x := 12;
      SiteSize.y := 12;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 10;
      Outline[0].x := -64;
      Outline[0].y := -50;
      Outline[1].x := -64;
      Outline[1].y := 465;
      Outline[2].x := -32;
      Outline[2].y := 527;
      Outline[3].x := -22;
      Outline[3].y := 589;
      Outline[4].x := 0;
      Outline[4].y := 639;
      Outline[5].x := 22;
      Outline[5].y := 589;
      Outline[6].x := 32;
      Outline[6].y := 527;
      Outline[7].x := 64;
      Outline[7].y := 465;
      Outline[8].x := 64;
      Outline[8].y := -50;
      Outline[9].x := Outline[0].x;
      Outline[9].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[4].x := -27;
      SiteLoc[4].y := 155;
      SiteLoc[5].x := -27;
      SiteLoc[5].y := 93;
      SiteLoc[6].x := -27;
      SiteLoc[6].y := 217;
      SiteLoc[7].x := -27;
      SiteLoc[7].y := 341;
      SiteLoc[8].x := -27;
      SiteLoc[8].y := 31;
      SiteLoc[9].x := -27;
      SiteLoc[9].y := 279;
      SiteLoc[10].x := -27;
      SiteLoc[10].y := 403;
      SiteLoc[11].x := -27;
      SiteLoc[11].y := 465;
      SiteLoc[3].x := 27;
      SiteLoc[3].y := 436;
      SiteLoc[2].x := 27;
      SiteLoc[2].y := 372;
      SiteLoc[1].x := 27;
      SiteLoc[1].y := 248;
      SiteLoc[0].x := 27;
      SiteLoc[0].y := 0;
      SiteLoc[15].x := 27;
      SiteLoc[15].y := 310;
      SiteLoc[14].x := 27;
      SiteLoc[14].y := 186;
      SiteLoc[13].x := 27;
      SiteLoc[13].y := 62;
      SiteLoc[12].x := 27;
      SiteLoc[12].y := 124;
    end;


    // 16CHAN5 came from Steve Biere
    With KnownElectrode[4] do
    begin
      Name := '16CHAN5';
      Description := '16 Chan Silicon, 4 Shank';
      SiteSize.x := 10;
      SiteSize.y := 10;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 5;
      Outline[0].x := -165;
      Outline[0].y := -50;
      Outline[1].x := -165;
      Outline[1].y := 200;
      Outline[2].x := 164;
      Outline[2].y := 200;
      Outline[3].x := 164;
      Outline[3].y := -50;
      Outline[4].x := Outline[0].x;
      Outline[4].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[0].x := -50;
      SiteLoc[0].y := 0;
      SiteLoc[1].x := -50;
      SiteLoc[1].y := 50;
      SiteLoc[2].x := -50;
      SiteLoc[2].y := 100;
      SiteLoc[3].x := -50;
      SiteLoc[3].y := 150;
      SiteLoc[4].x := -150;
      SiteLoc[4].y := 150;
      SiteLoc[5].x := -150;
      SiteLoc[5].y := 100;
      SiteLoc[6].x := -150;
      SiteLoc[6].y := 50;
      SiteLoc[7].x := -150;
      SiteLoc[7].y := 0;
      SiteLoc[8].x := 50;
      SiteLoc[8].y := 0;
      SiteLoc[9].x := 50;
      SiteLoc[9].y := 50;
      SiteLoc[10].x := 50;
      SiteLoc[10].y := 100;
      SiteLoc[11].x := 50;
      SiteLoc[11].y := 150;
      SiteLoc[12].x := 150;
      SiteLoc[12].y := 150;
      SiteLoc[13].x := 150;
      SiteLoc[13].y := 100;
      SiteLoc[14].x := 150;
      SiteLoc[14].y := 50;
      SiteLoc[15].x := 150;
      SiteLoc[15].y := 0;
    end;

    //RX01 is 4 channel UMICH design
    //Not yet coded !
    With KnownElectrode[5] do
    begin
      Name := 'RX01';
      Description := '4 Chan Silicon, Linear';
      SiteSize.x := 12;
      SiteSize.y := 12;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 3;
      Outline[0].x := -64;
      Outline[0].y := -50;
      Outline[1].x := -64;
      Outline[1].y := 465;
      Outline[2].x := Outline[0].x;
      Outline[2].y := Outline[0].y;

      NumSites := 4;
      CenterX := 0;
      SiteLoc[0].x := -27;
      SiteLoc[0].y := 279;
      SiteLoc[1].x := -27;
      SiteLoc[1].y := 217;
      SiteLoc[2].x := -27;
      SiteLoc[2].y := 155;
      SiteLoc[3].x := -27;
      SiteLoc[3].y := 93;
    end;

    //RX02 is 4 channel UMICH design
    //Not yet coded !
    With KnownElectrode[6] do
    begin
      Name := 'RX02';
      Description := '4 Chan Silicon, Linear';
      SiteSize.x := 12;
      SiteSize.y := 12;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 3;
      Outline[0].x := -64;
      Outline[0].y := -50;
      Outline[1].x := -64;
      Outline[1].y := 465;
      Outline[2].x := Outline[0].x;
      Outline[2].y := Outline[0].y;

      NumSites := 4;
      CenterX := 0;
      SiteLoc[0].x := -27;
      SiteLoc[0].y := 279;
      SiteLoc[1].x := -27;
      SiteLoc[1].y := 217;
      SiteLoc[2].x := -27;
      SiteLoc[2].y := 155;
      SiteLoc[3].x := -27;
      SiteLoc[3].y := 93;
    end;

    //16CHAN01 is 16 channel UMICH design 16 channel linear
    With KnownElectrode[7] do
    begin
      Name := '16CHAN3';
      Description := '16 Chan Silicon, Linear';
      SiteSize.x := 5;
      SiteSize.y := 15;
      RoundSite := FALSE;
      Created := FALSE;

      NumPoints := 6;
      Outline[0].x := -10;
      Outline[0].y := -50;
      Outline[1].x := -10;
      Outline[1].y := 1510;
      Outline[2].x := 0;
      Outline[2].y := 1600;
      Outline[3].x := 10;
      Outline[3].y := 1510;
      Outline[4].x := 10;
      Outline[4].y := -50;
      Outline[5].x := Outline[0].x;
      Outline[5].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[0].x := 0;
      SiteLoc[0].y := 600;
      SiteLoc[1].x := 0;
      SiteLoc[1].y := 400;
      SiteLoc[2].x := 0;
      SiteLoc[2].y := 200;
      SiteLoc[3].x := 0;
      SiteLoc[3].y := 0;
      SiteLoc[4].x := 0;
      SiteLoc[4].y := 800;
      SiteLoc[5].x := 0;
      SiteLoc[5].y := 1000;
      SiteLoc[6].x := 0;
      SiteLoc[6].y := 1200;
      SiteLoc[7].x := 0;
      SiteLoc[7].y := 1400;
      SiteLoc[8].x := 0;
      SiteLoc[8].y := 1500;
      SiteLoc[9].x := 0;
      SiteLoc[9].y := 1300;
      SiteLoc[10].x := 0;
      SiteLoc[10].y := 1100;
      SiteLoc[11].x := 0;
      SiteLoc[11].y := 900;
      SiteLoc[12].x := 0;
      SiteLoc[12].y := 100;
      SiteLoc[13].x := 0;
      SiteLoc[13].y := 300;
      SiteLoc[14].x := 0;
      SiteLoc[14].y := 500;
      SiteLoc[15].x := 0;
      SiteLoc[15].y := 700;
    end;

    //TET2X2 is UMICH design 4 tetrodes in a 2X2 layout
    With KnownElectrode[8] do
    begin
      Name := 'TET2X2';
      Description := '16 Chan Silicon, 2 Shank';
      SiteSize.x := 13;
      SiteSize.y := 13;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 11;
      Outline[0].x := -115;
      Outline[0].y := -50;
      Outline[1].x := 115;
      Outline[1].y := -50;
      Outline[2].x := 115;
      Outline[2].y := 200;
      Outline[3].x := 75;
      Outline[3].y := 235;
      Outline[4].x := 35;
      Outline[4].y := 200;
      Outline[5].x := 35;
      Outline[5].y := -48;
      Outline[6].x := -35;
      Outline[6].y := -48;
      Outline[7].x := -35;
      Outline[7].y := 200;
      Outline[8].x := -75;
      Outline[8].y := 235;
      Outline[9].x := -115;
      Outline[9].y := 200;
      Outline[10].x := Outline[0].x;
      Outline[10].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[0].x := -75;
      SiteLoc[0].y := 150;
      SiteLoc[1].x := -75;
      SiteLoc[1].y := 0;
      SiteLoc[2].x := -100;
      SiteLoc[2].y := 25;
      SiteLoc[3].x := -75;
      SiteLoc[3].y := 200;
      SiteLoc[4].x := -75;
      SiteLoc[4].y := 50;
      SiteLoc[5].x := -100;
      SiteLoc[5].y := 175;
      SiteLoc[6].x := -50;
      SiteLoc[6].y := 25;
      SiteLoc[7].x := -50;
      SiteLoc[7].y := 175;
      SiteLoc[8].x := 50;
      SiteLoc[8].y := 175;
      SiteLoc[9].x := 50;
      SiteLoc[9].y := 25;
      SiteLoc[10].x := 100;
      SiteLoc[10].y := 175;
      SiteLoc[11].x := 75;
      SiteLoc[11].y := 0;
      SiteLoc[12].x := 75;
      SiteLoc[12].y := 150;
      SiteLoc[13].x := 100;
      SiteLoc[13].y := 25;
      SiteLoc[14].x := 75;
      SiteLoc[14].y := 50;
      SiteLoc[15].x := 75;
      SiteLoc[15].y := 200;
    end;

    //TET2X2X is UMICH design 4 tetrodes in a 2X2 layout, but with PreAmp banks swapped
    With KnownElectrode[9] do
    begin
      Name := 'TET2X2X';
      Description := '16 Chan Silicon, 2 Shank, Banks Crossed';
      SiteSize.x := 13;
      SiteSize.y := 13;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 11;
      Outline[0].x := -115;
      Outline[0].y := -50;
      Outline[1].x := 115;
      Outline[1].y := -50;
      Outline[2].x := 115;
      Outline[2].y := 200;
      Outline[3].x := 75;
      Outline[3].y := 235;
      Outline[4].x := 35;
      Outline[4].y := 200;
      Outline[5].x := 35;
      Outline[5].y := -48;
      Outline[6].x := -35;
      Outline[6].y := -48;
      Outline[7].x := -35;
      Outline[7].y := 200;
      Outline[8].x := -75;
      Outline[8].y := 235;
      Outline[9].x := -115;
      Outline[9].y := 200;
      Outline[10].x := Outline[0].x;
      Outline[10].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[0].x := 50;
      SiteLoc[0].y := 175;
      SiteLoc[1].x := 50;
      SiteLoc[1].y := 25;
      SiteLoc[2].x := 100;
      SiteLoc[2].y := 175;
      SiteLoc[3].x := 75;
      SiteLoc[3].y := 0;
      SiteLoc[4].x := 75;
      SiteLoc[4].y := 150;
      SiteLoc[5].x := 100;
      SiteLoc[5].y := 25;
      SiteLoc[6].x := 75;
      SiteLoc[6].y := 50;
      SiteLoc[7].x := 75;
      SiteLoc[7].y := 200;
      SiteLoc[8].x := -75;
      SiteLoc[8].y := 150;
      SiteLoc[9].x := -75;
      SiteLoc[9].y := 0;
      SiteLoc[10].x := -100;
      SiteLoc[10].y := 25;
      SiteLoc[11].x := -75;
      SiteLoc[11].y := 200;
      SiteLoc[12].x := -75;
      SiteLoc[12].y := 50;
      SiteLoc[13].x := -100;
      SiteLoc[13].y := 175;
      SiteLoc[14].x := -50;
      SiteLoc[14].y := 25;
      SiteLoc[15].x := -50;
      SiteLoc[15].y := 175;
    end;

    //TETRODE is 4 channel tetrode of any arrangement
    With KnownElectrode[10] do
    begin
      Name := 'TETRODE';
      Description := 'Any Tetrode';
      SiteSize.x := 12;
      SiteSize.y := 12;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 5;
      Outline[0].x := -50;
      Outline[0].y := -50;
      Outline[1].x := -50;
      Outline[1].y := 200;
      Outline[2].x := 50;
      Outline[2].y := 200;
      Outline[3].x := 50;
      Outline[3].y := -50;
      Outline[4].x := Outline[0].x;
      Outline[4].y := Outline[0].y;

      NumSites := 4;
      CenterX := 0;
      SiteLoc[0].x := 0;
      SiteLoc[0].y := 0;
      SiteLoc[1].x := 0;
      SiteLoc[1].y := 50;
      SiteLoc[2].x := 0;
      SiteLoc[2].y := 100;
      SiteLoc[3].x := 0;
      SiteLoc[3].y := 150;
    end;

    //STEREOTRODE is 2 channel stereotrode of any arrangement
    With KnownElectrode[11] do
    begin
      Name := 'STEREOTRODE';
      Description := 'Any Stereotrode';
      SiteSize.x := 12;
      SiteSize.y := 12;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 5;
      Outline[0].x := -50;
      Outline[0].y := -50;
      Outline[1].x := -50;
      Outline[1].y := 100;
      Outline[2].x := 50;
      Outline[2].y := 100;
      Outline[3].x := 50;
      Outline[3].y := -50;
      Outline[4].x := Outline[0].x;
      Outline[4].y := Outline[0].y;

      NumSites := 2;
      CenterX := 0;
      SiteLoc[0].x := 0;
      SiteLoc[0].y := 0;
      SiteLoc[1].x := 0;
      SiteLoc[1].y := 50;
    end;

    //SINGLE is 1 channel electrode
    With KnownElectrode[12] do
    begin
      Name := 'SINGLE';
      Description := 'Any Single Electrode';
      SiteSize.x := 12;
      SiteSize.y := 12;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 5;
      Outline[0].x := -50;
      Outline[0].y := -50;
      Outline[1].x := -50;
      Outline[1].y := 50;
      Outline[2].x := 50;
      Outline[2].y := 50;
      Outline[3].x := 50;
      Outline[3].y := -50;
      Outline[4].x := Outline[0].x;
      Outline[4].y := Outline[0].y;

      NumSites := 1;
      CenterX := 0;
      SiteLoc[0].x := 0;
      SiteLoc[0].y := 0;
    end;

    //MMAP16_1a is TIMPAH test design
    With KnownElectrode[13] do
    begin
      Name := 'MMAP16_1a';
      Description := '16 Chan, 3 Col, 65µm spacing';
      SiteSize.x := 15;
      SiteSize.y := 15;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 9;
      Outline[0].x := -100;
      Outline[0].y := 0;
      Outline[1].x := -100;
      Outline[1].y := 445;
      Outline[2].x := -50;
      Outline[2].y := 565;
      Outline[3].x := -32;
      Outline[3].y := 627;
      Outline[4].x := 32;
      Outline[4].y := 627;
      Outline[5].x := 50;
      Outline[5].y := 565;
      Outline[6].x := 100;
      Outline[6].y := 445;
      Outline[7].x := 100;
      Outline[7].y := 0;
      Outline[8].x := Outline[0].x;
      Outline[8].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[0].x := 0;
      SiteLoc[0].y := 417;
      SiteLoc[1].x := -56;
      SiteLoc[1].y := 384;
      SiteLoc[2].x := 0;
      SiteLoc[2].y := 352;
      SiteLoc[3].x := 56;
      SiteLoc[3].y := 384;
      SiteLoc[4].x := -56;
      SiteLoc[4].y := 319;
      SiteLoc[5].x := 0;
      SiteLoc[5].y := 287;
      SiteLoc[6].x := 56;
      SiteLoc[6].y := 319;
      SiteLoc[7].x := -56;
      SiteLoc[7].y := 254;
      SiteLoc[8].x := 0;
      SiteLoc[8].y := 222;
      SiteLoc[9].x := 56;
      SiteLoc[9].y := 254;
      SiteLoc[10].x := -56;
      SiteLoc[10].y := 189;
      SiteLoc[11].x := 0;
      SiteLoc[11].y := 157;
      SiteLoc[12].x := 56;
      SiteLoc[12].y := 189;
      SiteLoc[13].x := -56;
      SiteLoc[13].y := 124;
      SiteLoc[14].x := 0;
      SiteLoc[14].y := 92;
      SiteLoc[15].x := 56;
      SiteLoc[15].y := 124;
    end;

    //MMAP16_1b is TIMPAH test design
    With KnownElectrode[14] do
    begin
      Name := 'MMAP16_1b';
      Description := 'Non-staggered, 3 Col';
      SiteSize.x := 15;
      SiteSize.y := 15;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 9;
      Outline[0].x := -100;
      Outline[0].y := 0;
      Outline[1].x := -100;
      Outline[1].y := 445;
      Outline[2].x := -50;
      Outline[2].y := 565;
      Outline[3].x := -32;
      Outline[3].y := 627;
      Outline[4].x := 32;
      Outline[4].y := 627;
      Outline[5].x := 50;
      Outline[5].y := 565;
      Outline[6].x := 100;
      Outline[6].y := 445;
      Outline[7].x := 100;
      Outline[7].y := 0;
      Outline[8].x := Outline[0].x;
      Outline[8].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[0].x := -43;
      SiteLoc[0].y := 417;
      SiteLoc[1].x := 0;
      SiteLoc[1].y := 417;
      SiteLoc[2].x := 43;
      SiteLoc[2].y := 417;
      SiteLoc[3].x := -43;
      SiteLoc[3].y := 367;
      SiteLoc[4].x := 0;
      SiteLoc[4].y := 367;
      SiteLoc[5].x := 43;
      SiteLoc[5].y := 367;
      SiteLoc[6].x := -43;
      SiteLoc[6].y := 317;
      SiteLoc[7].x := 0;
      SiteLoc[7].y := 317;
      SiteLoc[8].x := 43;
      SiteLoc[8].y := 317;
      SiteLoc[9].x := -43;
      SiteLoc[9].y := 267;
      SiteLoc[10].x := 0;
      SiteLoc[10].y := 267;
      SiteLoc[11].x := 43;
      SiteLoc[11].y := 267;
      SiteLoc[12].x := -43;
      SiteLoc[12].y := 217;
      SiteLoc[13].x := 0;
      SiteLoc[13].y := 217;
      SiteLoc[14].x := 43;
      SiteLoc[14].y := 217;
      SiteLoc[15].x := 0;
      SiteLoc[15].y := 167;
    end;

    //MMAP16_1c is TIMPAH test design
    With KnownElectrode[15] do
    begin
      Name := 'MMAP16_1c';
      Description := '3 Col, Staggered, 75µm spacing';
      SiteSize.x := 15;
      SiteSize.y := 15;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 9;
      Outline[0].x := -100;
      Outline[0].y := 0;
      Outline[1].x := -100;
      Outline[1].y := 445;
      Outline[2].x := -50;
      Outline[2].y := 565;
      Outline[3].x := -32;
      Outline[3].y := 627;
      Outline[4].x := 32;
      Outline[4].y := 627;
      Outline[5].x := 50;
      Outline[5].y := 565;
      Outline[6].x := 100;
      Outline[6].y := 445;
      Outline[7].x := 100;
      Outline[7].y := 0;
      Outline[8].x := Outline[0].x;
      Outline[8].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[0].x := 0;
      SiteLoc[0].y := 417;
      SiteLoc[1].x := -65;
      SiteLoc[1].y := 379;
      SiteLoc[2].x := 0;
      SiteLoc[2].y := 342;
      SiteLoc[3].x := 65;
      SiteLoc[3].y := 379;
      SiteLoc[4].x := -65;
      SiteLoc[4].y := 304;
      SiteLoc[5].x := 0;
      SiteLoc[5].y := 267;
      SiteLoc[6].x := 65;
      SiteLoc[6].y := 304;
      SiteLoc[7].x := -65;
      SiteLoc[7].y := 229;
      SiteLoc[8].x := 0;
      SiteLoc[8].y := 192;
      SiteLoc[9].x := 65;
      SiteLoc[9].y := 229;
      SiteLoc[10].x := -65;
      SiteLoc[10].y := 154;
      SiteLoc[11].x := 0;
      SiteLoc[11].y := 117;
      SiteLoc[12].x := 65;
      SiteLoc[12].y := 154;
      SiteLoc[13].x := -65;
      SiteLoc[13].y := 79;
      SiteLoc[14].x := 0;
      SiteLoc[14].y := 42;
      SiteLoc[15].x := 65;
      SiteLoc[15].y := 79;
    end;

    //MMAP16_1aT is TIMPAH test design
    With KnownElectrode[16] do
    begin
      Name := 'MMAP16_1aT';
      Description := '16 Chan, 3 Col, top probe sites';
      SiteSize.x := 15;
      SiteSize.y := 15;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 5;
      Outline[0].x := -100;
      Outline[0].y := 0;
      Outline[1].x := -100;
      Outline[1].y := 445;
      Outline[2].x := 100;
      Outline[2].y := 445;
      Outline[3].x := 100;
      Outline[3].y := 0;
      Outline[4].x := Outline[0].x;
      Outline[4].y := Outline[0].y;

      NumSites := 16;
      CenterX := 0;
      SiteLoc[0].x := 0;
      SiteLoc[0].y := 417;
      SiteLoc[1].x := -56;
      SiteLoc[1].y := 384;
      SiteLoc[2].x := 0;
      SiteLoc[2].y := 352;
      SiteLoc[3].x := 56;
      SiteLoc[3].y := 384;
      SiteLoc[4].x := -56;
      SiteLoc[4].y := 319;
      SiteLoc[5].x := 0;
      SiteLoc[5].y := 287;
      SiteLoc[6].x := 56;
      SiteLoc[6].y := 319;
      SiteLoc[7].x := -56;
      SiteLoc[7].y := 254;
      SiteLoc[8].x := 0;
      SiteLoc[8].y := 222;
      SiteLoc[9].x := 56;
      SiteLoc[9].y := 254;
      SiteLoc[10].x := -56;
      SiteLoc[10].y := 189;
      SiteLoc[11].x := 0;
      SiteLoc[11].y := 157;
      SiteLoc[12].x := 56;
      SiteLoc[12].y := 189;
      SiteLoc[13].x := -56;
      SiteLoc[13].y := 124;
      SiteLoc[14].x := 0;
      SiteLoc[14].y := 92;
      SiteLoc[15].x := 56;
      SiteLoc[15].y := 124;
    end;
end;

Function GetElectrode(var Electrode : TElectrode; Name : ShortString) : boolean;
var i : integer;
begin
   GetElectrode := FALSE;
   For i := 0 to KNOWNELECTRODES-1 do
     if Name = KnownElectrode[i].Name then
     begin
       Move(KnownElectrode[i],Electrode,sizeof(TElectrode));
       GetElectrode := TRUE;
     end;
end;

Initialization
  MakeKnownElectrodes;

end.
