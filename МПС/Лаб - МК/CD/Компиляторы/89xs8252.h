/*header file for the ATMEL microcontroller
	AT89LS8252
	AT89S8252
 */

#ifndef __89XS8252_H

#define __89XS8252_H
	
typedef struct {
	unsigned B7:1;	/* this is the most significant bit */
	unsigned B6:1;
	unsigned B5:1;
	unsigned B4:1;
	unsigned B3:1;
	unsigned B2:1;
	unsigned B1:1;
	unsigned B0:1;
}	SFR_BITS;

static volatile unsigned char		P0	@ 0x80;
static volatile unsigned char		SP	@ 0x81;
static volatile unsigned char		DP0L	@ 0x82;
static volatile unsigned char		DP0H	@ 0x83;
static volatile unsigned char		DP1L	@ 0x84;
static volatile unsigned char		DP1H	@ 0x85;
static volatile unsigned char		SPDR	@ 0x86; //SPI Data Register
static 		unsigned char		PCON	@ 0x87;
static volatile	unsigned char		TCON	@ 0x88;
static 		unsigned char		TMOD	@ 0x89;
static volatile unsigned char		TL0	@ 0x8A;
static volatile unsigned char		TL1	@ 0x8B;
static volatile unsigned char		TH0	@ 0x8C;
static volatile unsigned char		TH1	@ 0x8D;
static volatile unsigned char		P1	@ 0x90;
static volatile unsigned char		WMCON	@ 0x96;
static volatile	unsigned char		SCON	@ 0x98;
static volatile unsigned char		SBUF	@ 0x99;
static volatile unsigned char		P2	@ 0xA0;
static 		unsigned char		IE	@ 0xA8;
static volatile	unsigned char		SPSR	@ 0xAA;
static volatile unsigned char		P3	@ 0xB0;
static 		unsigned char		IP	@ 0xB8;
static volatile	unsigned char		T2CON	@ 0xC8;
static 		unsigned char		T2MOD	@ 0xC9;
static volatile unsigned char		RCAP2L	@ 0xCA;
static volatile unsigned char		RCAP2H	@ 0xCB;
static volatile unsigned char		TL2	@ 0xCC;
static volatile unsigned char		TH2	@ 0xCD;
static volatile unsigned char		PSW	@ 0xD0;
static		unsigned char		SPCR	@ 0xD5;
static volatile unsigned char		ACC	@ 0xE0;
static volatile unsigned char		B	@ 0xF0;

static volatile SFR_BITS	P0_BITS		@ 0x80;
static 		SFR_BITS	PCON_BITS	@ 0x87;
static volatile	SFR_BITS	TCON_BITS	@ 0x88;
static 		SFR_BITS	TMOD_BITS	@ 0x89;
static volatile SFR_BITS	P1_BITS		@ 0x90;
static volatile SFR_BITS	WMCON_BITS	@ 0x96;
static volatile	SFR_BITS	SCON_BITS	@ 0x98;
static volatile SFR_BITS	P2_BITS		@ 0xA0;
static volatile SFR_BITS	SPSR_BITS	@ 0xAA;
static 		SFR_BITS	IE_BITS		@ 0xA8;
static volatile SFR_BITS	P3_BITS		@ 0xB0;
static 		SFR_BITS	IP_BITS		@ 0xB8;
static volatile	SFR_BITS	T2CON_BITS	@ 0xC8;
static 		SFR_BITS	T2MOD_BITS	@ 0xC9;
static volatile SFR_BITS	PSW_BITS	@ 0xD0;
static volatile SFR_BITS	SPCR_BITS	@ 0xD5;
static volatile SFR_BITS	ACC_BITS	@ 0xE0;
static volatile SFR_BITS	B_BITS		@ 0xF0;

/*bit addressable locations*/
//Port 0 P0
static volatile bit	P00	@ 0x80;
static volatile bit	P01	@ 0x81;
static volatile bit	P02	@ 0x82;
static volatile bit	P03	@ 0x83;
static volatile bit	P04	@ 0x84;
static volatile bit	P05	@ 0x85;
static volatile bit	P06	@ 0x86;
static volatile bit	P07	@ 0x87;
// Alternate Functions
static volatile bit	AD0	@ 0x80;
static volatile bit	AD1	@ 0x81;
static volatile bit	AD2	@ 0x82;
static volatile bit	AD3	@ 0x83;
static volatile bit	AD4	@ 0x84;
static volatile bit	AD5	@ 0x85;
static volatile bit	AD6	@ 0x86;
static volatile bit	AD7	@ 0x87;

//Timer Control TCON
static 		bit	IT0	@ 0x88;
static volatile bit	IE0	@ 0x89;
static 		bit	IT1	@ 0x8A;
static volatile bit	IE1	@ 0x8B;
static 		bit	TR0	@ 0x8C;
static volatile bit	TF0	@ 0x8D;
static 		bit	TR1	@ 0x8E;
static volatile bit	TF1	@ 0x8F;

//Port 1 P1
static volatile bit	P10	@ 0x90;
static volatile bit	P11	@ 0x91;
static volatile bit	P12	@ 0x92;
static volatile bit	P13	@ 0x93;
static volatile bit	P14	@ 0x94;
static volatile bit	P15	@ 0x95;
static volatile bit	P16	@ 0x96;
static volatile bit	P17	@ 0x97;
// Alternate Functions
static volatile bit	T2	@ 0x90;
static volatile bit	T2EX	@ 0x91;
static volatile bit	SS	@ 0x94;
static volatile bit	MOSI	@ 0x95;
static volatile bit	MISO	@ 0x96;
static volatile bit	SCK	@ 0x97;

//Serial Control SCON
static volatile bit	RI	@ 0x98;
static volatile bit	TI	@ 0x99;
static volatile bit	RB8	@ 0x9A;
static		bit	TB8	@ 0x9B;
static 		bit	REN	@ 0x9C;
static		bit	SM2	@ 0x9D;
static		bit	SM1	@ 0x9E;
static		bit	SM0	@ 0x9F;

//Port 2 P2
static volatile bit	P20	@ 0xA0;
static volatile bit	P21	@ 0xA1;
static volatile bit	P22	@ 0xA2;
static volatile bit	P23	@ 0xA3;
static volatile bit	P24	@ 0xA4;
static volatile bit	P25	@ 0xA5;
static volatile bit	P26	@ 0xA6;
static volatile bit	P27	@ 0xA7;
// Alternate Functions
static volatile bit	A8	@ 0xA0;
static volatile bit	A9	@ 0xA1;
static volatile bit	A10	@ 0xA2;
static volatile bit	A11	@ 0xA3;
static volatile bit	A12	@ 0xA4;
static volatile bit	A13	@ 0xA5;
static volatile bit	A14	@ 0xA6;
static volatile bit	A15	@ 0xA7;

//Interrupt Enable IE
static		bit	EX0	@ 0xA8;
static		bit	ET0	@ 0xA9;
static		bit	EX1	@ 0xAA;
static		bit	ET1	@ 0xAB;
static 		bit	ES	@ 0xAC;
static 		bit	ET2	@ 0xAD;
static 		bit	EA	@ 0xAF;

//Port 3 P3
static volatile bit	P30	@ 0xB0;
static volatile bit	P31	@ 0xB1;
static volatile bit	P32	@ 0xB2;
static volatile bit	P33	@ 0xB3;
static volatile bit	P34	@ 0xB4;
static volatile bit	P35	@ 0xB5;
static volatile bit	P36	@ 0xB6;
static volatile bit	P37	@ 0xB7;
// Alternate Functions
static volatile bit	RXD	@ 0xB0;
static volatile bit	TXD	@ 0xB1;
static volatile bit	INT0	@ 0xB2;
static volatile bit	INT1	@ 0xB3;
static volatile bit	T0	@ 0xB4;
static volatile bit	T1	@ 0xB5;
static volatile bit	WR	@ 0xB6;
static volatile bit	RD	@ 0xB7;

//Interrupt Priority
static 		bit	PX0	@ 0xB8;
static 		bit	PT0	@ 0xB9;
static 		bit	PX1	@ 0xBA;
static 		bit	PT1	@ 0xBB;
static 		bit	PS	@ 0xBC;
static 		bit	PT2	@ 0xBD;

//Timer 2 Control T2CON
static 		bit	CPRL2	@ 0xC8;
static 		bit	CT2	@ 0xC9;
static 		bit	TR2	@ 0xCA;
static 		bit	EXEN2	@ 0xCB;
static volatile	bit	TCLK	@ 0xCC;
static volatile	bit	RCLK	@ 0xCD;
static volatile	bit	EXF2	@ 0xCE;
static volatile	bit	TF2	@ 0xCF;

//Program Status Word PSW
static volatile bit	P	@ 0xD0;
static volatile bit	F1	@ 0xD1;
static volatile bit	OV	@ 0xD2;
static 		bit	RS0	@ 0xD3;
static 		bit	RS1	@ 0xD4;
static volatile bit	F0	@ 0xD5;
static volatile bit	AC	@ 0xD6;
static volatile bit	CY	@ 0xD7;

//Accumulator ACC
static volatile bit	ACC0	@ 0xE0;
static volatile bit	ACC1	@ 0xE1;
static volatile bit	ACC2	@ 0xE2;
static volatile bit	ACC3	@ 0xE3;
static volatile bit	ACC4	@ 0xE4;
static volatile bit	ACC5	@ 0xE5;
static volatile bit	ACC6	@ 0xE6;
static volatile bit	ACC7	@ 0xE7;

//B register
static volatile bit	B0	@ 0xF0;
static volatile bit	B1	@ 0xF1;
static volatile bit	B2	@ 0xF2;
static volatile bit	B3	@ 0xF3;
static volatile bit	B4	@ 0xF4;
static volatile bit	B5	@ 0xF5;
static volatile bit	B6	@ 0xF6;
static volatile bit	B7	@ 0xF7;


/*	Interrupt vectors */
#define	RESET	0
#define	XT0	0x03
#define TM0	0x0B
#define	XT1	0x13
#define	TM1	0x1B
#define	S0	0x23
#define	TM2	0x2B

#endif
