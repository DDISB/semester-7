	Tera Term Pro version 2.0
	for MS-Windows 95/NT
	T. Teranishi Jun 7, 1996

	Copyright (C) 1994-1996 T. Teranishi
	All Rights Reserved.

Index

  1. About Tera Term
  2. Copyright and Notice
  3. Requirements
  4. Distribution Package
  5. Installation
  6. Usage
  7. Known Problems
  8. How to Get the Latest Version
  9. Acknowledgment

-------------------------------------------------------------------------------
1. About Tera Term

Tera Term (Pro) is a free software terminal emulator (communication program)
which supports:

	- Serial port connections.
	- TCP/IP (telnet) connections.
	- VT100 emulation, and selected VT200/300 emulation.
	- TEK4010 emulation.
	- File transfer protocols (Kermit, XMODEM, ZMODEM, B-PLUS,
	  and Quick-VAN).
	- Scripts using the "Tera Term Language".

-------------------------------------------------------------------------------
2. Copyright and Notice

Tera Term (Pro) is free software.

There is no warranty for damages caused by using this application.

Without written permission by the author (Takashi Teranishi), you may
not distribute modified packages of Tera Term, and may not distribute
Tera Term for profit.

For requests, questions, and bug reports, contact the author by e-mail
at the following address:

	teranishi@rikvax.riken.go.jp

-------------------------------------------------------------------------------
3. Requirements

1) Software

Supported operating systems:

	MS-Windows NT
	MS-Windows 95

	Note: Tera Term Pro is a 32-bit application.
	      For Windows 3.1, use Tera Term ver. 1.X.
	      TTMACRO.EXE and KEYCODE, included in the distribution package,
	      are 16-bit executable files.

2) Hardware

A modem or an ethernet board is required.

-------------------------------------------------------------------------------
4. Distribution Package

The distribution package contains the following files:

README.TXT	This document
READMEJ.TXT	Japanese version of README.TXT
CMNDLINE.TXT	Description of the command line format
TTERMPRO.EXE	Tera Term Pro executable file
TTPCMN.DLL	Dynamic link library for Tera Term Pro
TTPDLG.DLL	Dynamic link library for Tera Term Pro
TTPFILE.DLL	Dynamic link library for Tera Term Pro
TTPSET.DLL	Dynamic link library for Tera Term Pro
TTPTEK.DLL	Dynamic link library for Tera Term Pro
TSPECIAL.TTF	Special font for Tera Term (True Type)
TTERMP.HLP	Help file
TTERMPJ.HLP	Japanese version of TTERM.HLP
TERATERM.INI	Tera Term setup file
IBMKEYB.CNF	Sample keyboard setup file for the IBM-PC/AT 101-key keyboard
PC98KEYB.CNF	Sample keyboard setup file for the NEC PC98 keyboard (Win 95)
NT98KEYB.CNF	Sample keyboard setup file for the NEC PC98 keyboard (Win NT)
KEYCODE.EXE	Utility to display key codes
KEYCODE.TXT	Description of KEYCODE.EXE
KEYCODEJ.TXT	Japanese version of KEYCODE.TXT
TTMACRO.EXE	Macro interpreter
TTMACRO.TXT	Description of TTMACRO
TTMACROJ.TXT	Japanese version of TTMACRO.TXT
DIALUP.TTL	Sample macro file (dial-up login by modem)
LOGIN.TTL	Sample macro file (auto login by telnet)

-------------------------------------------------------------------------------
5. Installation

1) Copy all files included in the distribution package to an empty directory
(folder) of your choice.

2) If you are using Windows NT, install TTERMPRO.EXE in Program Manager
with its icon. If you are using Windows 95, create a shortcut
for TTERMPRO.EXE in a folder of your choice, or in the Start menu,
or on the desktop.

If you want to connect to a host by telnet, the host name can be specified
as a parameter in the command line (shortcut link). See CMNDLINE.TXT.
If you want to connect to a host by a serial port, command line parameters
can be omitted.

Specify "Working directory" as the directory in which TTERMPRO.EXE
exists.

3) To use the DEC special font, install the font TSPECIAL.TTF in Windows
by using Control Panel. When you uninstall or upgrade Tera Term later,
unistall the font using Control Panel. This font is used automatically
by Tera Term. You can not select it in the [Setup] font dialog box.

-------------------------------------------------------------------------------
6. Usage

If you have a previous version of Tera Term, note that this version of
Tera Term and the previous one can not run simultaneously. Close all old
Tera Term sessions before running the new Tera Term session.
Re-using old setup files with this version of Tera Term is not recommended.

The first time you run Tera Term, the General Setup dialog box appears.
Choose the port type ("TCP/IP" or "Serial") you mainly use, and the language
type "English". Click the Ok button, then the Tera Term window appears.

If you choose the port type as "TCP/IP", the New connection dialog box
(to enter the host name to be connected) is displayed after the Tera Term
window appears. If you want to change the setup before the first connection
is started, click the Cancel button here.

You can view the help file for usage, by using the [Help] Index command.

See KEYCODE.TXT for a description of the keyboard setup.
See TTMACRO.TXT for a description of scripts.

-------------------------------------------------------------------------------
7. Known Problems

TTMACRO.EXE and KEYCODE.EXE, included in the distribution package,
are 16-bit executable files. Long macro filenames are not supported.

The Kermit Send, ZMODEM Send, and Quick-VAN Send dialog boxes do not support
filenames which contain the space characters, such as "abc def.dat".
They are displayed in DOS format, like "abcdef~.dat".

-------------------------------------------------------------------------------
8. How to Get the Latest Version

You may find the latest version of Tera Term at the following ftp sites:

ftp://riksun.riken.go.jp/pub/pc/misc/terminal/teraterm/
ftp://utsun.s.u-tokyo.ac.jp/PC/terminal/teraterm/

-------------------------------------------------------------------------------
9. Acknowledgment

I would like to thank everyone who sent bug reports and suggestions.
I especially thank the people who have supported the development
of Tera Term from very early on. I also wish to thank Mr. Luigi M Bianchi 
for helping with the documentation included in the distribution package.