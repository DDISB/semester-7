@ECHO OFF
hex2bin test.hex program.bin
isp_hb.exe /LPT1 /ERASE program.bin
