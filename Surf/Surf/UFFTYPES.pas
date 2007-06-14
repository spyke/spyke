{ (c) 1994-98 Phil Hetherington, P&M Research Technologies, Inc.}
UNIT UFFTYPES;
INTERFACE
CONST
(* File Header definitions - denotes the size of fields in BYTES *)
   UFF_FH_REC_TYPE      = 1;
   UFF_DRDB_REC_TYPE    = 2;
   UFF_MIN_REC_TYPE     = 3;

   UFF_FILEHEADER_LEN	= 2048;
   UFF_NAME_LEN	    	= 10;
   UFF_OSNAME_LEN       = 12;
   UFF_NODENAME_LEN  	= 32;
   UFF_DEVICE_LEN       = 32;
   UFF_PATH_LEN	     	= 160;
   UFF_FILENAME_LEN  	= 32;
   UFF_FH_PAD_LEN       = 76;
   UFF_APPINFO_LEN   	= 32;
   UFF_OWNER_LEN        = 14;
   UFF_FILEDESC_LEN  	= 64;
   UFF_FH_USERAREA_LEN	= UFF_FILEHEADER_LEN - 512;

(* Data Record Descriptor Block definitions - field sizes in BYTES *)
   UFF_DRDB_BLOCK_LEN	 = 2048;
   UFF_DRDB_NAME_LEN	 = 20;
   UFF_DRDB_PAD_LEN	 = 16;
   UFF_DRDB_RSFD_NAME_LEN= 20;
   UFF_RSFD_PER_DRDB	 = 77;

TYPE
   TIMEDATE = record   (* reverse of C'S DATETIME *)
     Sec,Min,Hour,Day,Month,Year : WORD;
     junk : array[0..5] of BYTE;
   end;

   UFF_FILE_HEADER = record
      FH_rec_type                       : BYTE;  // 1 must be 1
      FH_rec_type_ext                   : BYTE;  // 1 must be 0
      UFF_name : array[0..UFF_NAME_LEN-1] of CHAR; // 10 must be "UFF" sz
      UFF_major                         : BYTE;  // 1 major UFF ver
      UFF_minor                         : BYTE;  // 1 minor UFF ver
      FH_rec_len                        : WORD;  // 2 FH record length in bytes
      DRDB_rec_len                      : WORD;  // 2 DBRD record length in bytes
      bi_di_seeks                       : WORDBOOL; // 2 bi-directional seeks format
      OS_name : array[0..UFF_OSNAME_LEN-1] of CHAR;  // 12 OS name ie "MS-DOS"
      OS_major                          : BYTE;  // 1 OS major rev
      OS_minor                          : BYTE;  // 1 OS minor rev
      create                            : TIMEDATE; // 18 creation time & date
      append                            : TIMEDATE; // 18 last append time & date
      node      : array[0..UFF_NODENAME_LEN-1]    of CHAR;  // 32 system node name - same as BDT
      device    : array[0..UFF_DEVICE_LEN-1]      of CHAR;  // 32 device name - same as BDT
      path      : array[0..UFF_PATH_LEN-1]        of CHAR;  // 160 path name
      filename  : array[0..UFF_FILENAME_LEN-1]    of CHAR;  // 32 original file name at creation
      pad       : array[0..UFF_FH_PAD_LEN-1]      of CHAR;  // 76 pad area to bring uff area to 512
      app_info  : array[0..UFF_APPINFO_LEN-1]     of CHAR;  // 32 application task name & version
      user_name : array[0..UFF_OWNER_LEN-1]       of CHAR;  // 14 user's name as owner of file
      file_desc : array[0..UFF_FILEDESC_LEN-1]    of CHAR;  // 64 description of file/exp
      user_area : array[0..UFF_FH_USERAREA_LEN-1] of BYTE;  // 1536 additional user area
      bd_FH_rec_type                    : BYTE;     // 1 record type; must be 1 BIDIRECTIONAL SUPPORT
      bd_FH_rec_type_ext                : BYTE;     // 1 record type extension; must be 0 BIDIRECTIONAL SUPPORT
                                          {total = 2050 bytes}
   end;

   drdbrsfname = array[1..UFF_DRDB_RSFD_NAME_LEN] of CHAR;

   UFF_DRDB_RSFD = record
{20}  subfield_name        : drdbrsfname; (* sz DRDB subfield name*)
{22}  subfield_data_type   : WORD;    (* underlying data type *)
{26}  subfield_size        : LONGINT; (* sub field size in bytes *)
   end;
{77*26=2002}
   UFF_DATA_REC_DESC_BLOCK = record
{1}   DRDB_rec_type                       : BYTE;    (* record type; must be 2 *)
{2}   DRDB_rec_type_ext                   : BYTE;    (* record type extension *)
{3}   DR_rec_type                         : CHAR;    (* Data Record type for DBRD 3-255 *)
{4}   DR_rec_type_ext                     : BYTE;    (* Data Record type ext; ignored *)
{8}   DR_size                             : LONGINT; (* Data Record size in bytes *)
{28}  DR_name:array[0..UFF_DRDB_NAME_LEN-1] of CHAR; (* Data Record name *)
{30}  DR_num_fields                       : WORD;    (* number of sub-fields in Data Record*)
{46}  DR_pad : array[0..UFF_DRDB_PAD_LEN-1] of BYTE; (* pad bytes for expansion *)
{2048}DR_subfields : array[1..UFF_RSFD_PER_DRDB] of UFF_DRDB_RSFD; (* sub fields desc *)
{2049}bd_DRDB_rec_type                     : BYTE;   (* record type; must be 2 BIDIRECTIONAL SUPPORT*)
{2050}bd_DRDB_rec_type_ext                 : BYTE;   (* record type extension; must be 0 BIDIRECTIONAL SUPPORT*)
   end;

IMPLEMENTATION
END.
