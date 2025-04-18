#pragma once

#define NUM_OF_MFCCS  13		// Amount of MFCCs in dataset
#define NUM_OF_CLUSTERS  8		// Amount of clusters (note: must be either n or n-1, where n is amount of genres) 

// Enum class with all genres
enum class e_Genres
{
	ALTERNATIVE_METAL	= 0,
	BLACK_METAL			= 1,
	DEATH_METAL			= 2,
	HEAVY_METAL			= 3,
	HARD_ROCK			= 4,
	NU_METAL			= 5,
	THRASH_METAL		= 6,
	PRIMUS				= 7,
};