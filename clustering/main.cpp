/*
Variables naming convention
---------------------------
Datatype			 Prefix
char					i1

short					i2
int						i4
long long				i8

unsigned short			u2
unsigned int			u4
unsigned long			u8

float					f4
doable					f8

enum					e

struct					s

class					c

std::string				str
std::array				a
std::vector				vec
*/

#include <stdio.h>
#include <iostream>

#include "c_KMeans.h" // k-means algotithm

int main()
{
	c_KMeans cAlg;
	cAlg.RunAlgorithm();

	return 0;
}