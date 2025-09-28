#include <89xs8252.h>
#include <intrpt.h>

#define r_led P00
#define y_led P01
#define g_led P02
#define b_led P03
#define lcd_led P04
#define DB P2
#define E P37
#define RS P36
#define RW P35

unsigned char rbr, ybr, gbr, bbr, sbr, ms, t1, t2, ms12_5, rg_st, lastchar, key, kr, kp;
unsigned char brightness[40] = {
	0, 0, 1, 1, 1, 2, 2, 2, 3, 3,
	4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
	9, 9, 10,11,12,13,14,15,16,17,
        18,19,20,21,23,26,29,32,35,39};

char buf2[16];

void ShowHex(unsigned char);

void wr_EEPROM(unsigned int addr,unsigned char data)
{
  while(!(WMCON&2));
  DP0L=addr;  //addr1.byte_.l_byte;
  DP0H=addr>>8;
  WMCON|=0x10;
  ACC =data;
  #asm
  movx @DPTR,A
  #endasm
  WMCON&=0xef;
}

unsigned char rd_EEPROM(unsigned int addr)
{
  while(!(WMCON&2));
  DP0L=addr;
  DP0H=addr>>8;
  #asm
  movx A,@DPTR
  #endasm
  return ACC;
}


void waitms(unsigned char m){
  unsigned char a;
  a = ms+(m<<1);
  while(ms!=a);
}

void wait12_5ms(unsigned char m){
  unsigned char a = ms12_5 + m;
  while(ms12_5!=a);
}

void outcw(unsigned char c){
unsigned char i;
unsigned int j;
  RS = 0;
  DB = c;
  E = 1;
  E = 2;
  for (i=0; i<20; i++);
  if (c==1||c==2||c==3)
    for (j=0; j<500; j++);
}
  
void outd(unsigned char c){
unsigned char i;
  RS = 1;
  DB = c;
  E = 1;
  E = 2;
  for (i=0; i<21; i++);
}

void clear1(){
  unsigned char i;
  outcw(0x80);
  for(i=0;i<14;i++)
    outd(' ');
  rg_st = t1 = 0;
  ShowHex(lastchar);
}

void clear2(){
  unsigned char i;
  outcw(0xC0);
  for(i=0;i<16;i++)
    outd(' ');
  t2 = 0;
}

void type1(unsigned char c){
  if (c==13||c==10) {clear1(); return;}
  if (t1>=13)
    clear1();
  outcw(t1|0x80);
  outd(c);
  t1++;
}

void type2(unsigned char c){
  if (c==13||c==10) {clear2(); return;}
  if(t2==16)
    clear2();
  if (c==8&&t2){
    t2--;
    outcw(t2|0xC0);
    outd(' ');
  } else {
    outcw(t2|0xC0);
    outd(c);
    buf2[t2] = c;
    ++t2;
  }
}

void init(){
unsigned char i;
//ROM_VECTORS
  ROM_VECTOR(TM0, int_timer0);
  ROM_VECTOR(XT0, int_XT0);
  ROM_VECTOR(XT1, int_XT1);
  ROM_VECTOR(S0, int_UART);
//Memory
  WMCON|=0x08;	// internal EEPROM enable
  WMCON&=0xfb;  // DPTR = DP0
//var
  if ((sbr = rd_EEPROM(0x123))>80) sbr = 80;
  t1=t2=0;
  rg_st = 0;
  lastchar=0;
  key = 0;
  kr = 0;
//UART
  PCON |= 0x80;	// SMOD=1
  SCON = 0x72;	// mode 1, receiver enable
  TMOD = 0x22;	//Timers 0&1 are 8-bit timers with auto-reload
  TH1   = 0xF5;	// 9600 baud at 20 MHz
  TR1   = 1;	// enable count 
  ES = 1;
//Timer 0
  TH0 = 89;
  TR0 = 1;
  ET0 = 1;
  EX0 = 1;
  EX1 = 1;
//  IT0 = 1;
  PT0 = 1;
  EA = 1;
//LCD
  waitms(30);
  RW = 0;
  outcw(0x3C);
  outcw(0x0C);
  outcw(0x01);
  outcw(0x06);
  outcw(0x40);
  for(i=0;i<8;i++) outd(0);
  for(i=0;i<8;i++) outd(0x10);
  for(i=0;i<8;i++) outd(0x18);
  for(i=0;i<8;i++) outd(0x1C);
  for(i=0;i<8;i++) outd(0x1E);
  for(i=0;i<8;i++) outd(0x1F);
}

void outbr(){
  unsigned char c = sbr,t=0;
  outcw(0x80);
  while(c>0){
    if(c>4){c-=5; outd(5); t++;}
    else {outd(c); c=0; t++;}
  }
  while(t++<16) outd(' ');
}

unsigned char translate(unsigned char c){
switch (c){
  case 'À': return 'A';
  case 'Á': return 0xA0;
  case 'Â': return 'B';
  case 'Ã': return 0xA1;
  case 'Ä': return 0xE0;
  case 'Å': return 'E';
  case '¨': return 0xA2;
  case 'Æ': return 0xA3;
  case 'Ç': return 0xA4;
  case 'È': return 0xA5;
  case 'É': return 0xA6;
  case 'Ë': return 0xA7;
  case 'Ì': return 'M';
  case 'Í': return 'H';
  case 'Î': return 'O';
  case 'Ï': return 0xA8;
  case 'Ð': return 'P';
  case 'Ñ': return 'C';
  case 'Ò': return 'T';
  case 'Ó': return 0xA9;
  case 'Ô': return 0xAA;
  case 'Õ': return 'X';
  case 'Ö': return 0xE1;
  case '×': return 0xAB;
  case 'Ø': return 0xAC;
  case 'Ù': return 0xE2;
  case 'Ú': return 0xAD;
  case 'Û': return 0xAE;
  case 'Ü': return 'b';
  case 'Ý': return 0xAF;
  case 'Þ': return 0xB0;
  case 'ß': return 0xB1;
  case 'à': return 'a';
  case 'á': return 0xB2;
  case 'â': return 0xB3;
  case 'ã': return 0xB4;
  case 'ä': return 0xE3;
  case 'å': return 'e';
  case '¸': return 0xB5;
  case 'æ': return 0xB6;
  case 'ç': return 0xB7;
  case 'è': return 0xB8;
  case 'é': return 0xB9;
  case 'ê': return 0xBA;
  case 'ë': return 0xBB;
  case 'ì': return 0xBC;
  case 'í': return 0xBD;
  case 'î': return 'o';
  case 'ï': return 0xBE;
  case 'ð': return 'p';
  case 'ñ': return 'c';
  case 'ò': return 0xBF;
  case 'ó': return 'y';
  case 'ô': return 0xE4;
  case 'õ': return 'x';
  case 'ö': return 0xE5;
  case '÷': return 0xC0;
  case 'ø': return 0xC1;
  case 'ù': return 0xE6;
  case 'ú': return 0xC2;
  case 'û': return 0xC3;
  case 'ü': return 0xC4;
  case 'ý': return 0xC5;
  case 'þ': return 0xC6;
  case 'ÿ': return 0xC7;
  default: return c;
  }
//  return 0;
}

void main(){
unsigned char i;
  init();
  if ((t2 = rd_EEPROM(0x200))>16) t2 = 0;
  outcw(0x0F);
  outcw(0xC0);
  for(i=0;i<t2;i++){
    buf2[i] = rd_EEPROM(0x201+i);
    outd(buf2[i]);
    wait12_5ms(16);
  }
  wait12_5ms(120);
  outcw(0x0C);
  while(1){
    if (key){
      if (rg_st) clear1();
      if (key!=13)
        type1(key);
      else
        clear1();
      SBUF = key;
      while(kp);
      key = 0;
    }
  }
}

void t2s(){
  if (rd_EEPROM(0x123)!=sbr)
    wr_EEPROM(0x123,sbr);
}

void t12_5ms(){
  unsigned char i;
  static unsigned char sled;
  if (++sled>=160) {sled = 0; t2s();}
  if(sled<40) {rbr = sled; bbr = 39-sled;}
  else if(sled<80) {ybr = sled - 40; rbr = 79-sled;}
  else if(sled<120) {gbr = sled - 80; ybr = 119 - sled;}
  else {bbr = sled - 120; gbr = 159-sled;}
  ms12_5++;
  if (rg_st&&ms12_5==rg_st)
    clear1(); 
  if(ms12_5&1) return;
//25 ms
  kp = 0;
  P0 = P0&0x1F|0xC0;
  for(i=0;i<10;i++);
  kr = !P13;
  if (!P10) kp = '7';
  else if (!P11) kp = '4';
  else if (!P12) kp = '1';
  else if (!P13) kp = '#';
  P0 = P0&0x1F|0xA0;
  for(i=0;i<10;i++);
  if (!P10) kp = '8';
  else if (!P11) kp = '5';
  else if (!P12) kp = '2';
  else if (!P13) kp = '0';
  P0 = P0&0x1F|0x60;
  for(i=0;i<10;i++);
  if (!P10) kp = '9';
  else if (!P11) kp = '6';
  else if (!P12) kp = '3';
  else if (!P13) kp = 13;
  if (kp) key = kp;
}

static interrupt void int_timer0(){
  static unsigned char tx,sa,st;
  if (++tx>4) tx = 0; else return;

  if (++sa>39) sa = 0;
  lcd_led = (sa>=(sbr>>1));

  r_led = (sa>=brightness[rbr]);
  y_led = (sa>=brightness[ybr]);
  g_led = (sa>=brightness[gbr]);
  b_led = (sa>=brightness[bbr]);
  
  if (++st>24){st = 0; t12_5ms(); }
  ms++;
}

static interrupt void int_XT0(){
  unsigned int i;
//  static unsigned char cc;
  for(i=0;i<1500;i++);
  if (sbr>0) sbr--;
//  type2(cc++);
  outbr();
  rg_st = ms12_5+80;
  if (!rg_st) rg_st++;
}

static interrupt void int_XT1(){
  unsigned int i;
  for(i=0;i<1500;i++);
  if (sbr<80) sbr++;
  outbr();
  rg_st = ms12_5+80;
  if (!rg_st) rg_st++;
}

void SaveBuf(){
  char i;
  wr_EEPROM(0x200,t2);
  for(i=0;i<t2;i++)
    wr_EEPROM(0x201+i,buf2[i]);
}

void ShowHex(unsigned char c){
  unsigned char b = c>>4;
  c &= 15;
  outcw(0x8E);
  if (b<10) b+='0';
  else b+='A'-10;
  if(c<10) c+='0';
  else c+='A'-10;
  outd(b);
  outd(c);
}

static interrupt void int_UART(){
  if (RI){
    if (SBUF==0x13) SaveBuf(); //Ctrl-S
    else 
      type2(translate(SBUF));
    ShowHex(SBUF);
    SBUF = SBUF;
    lastchar = SBUF;
    RI=0;
  }
  if (TI) TI = 0;
}
