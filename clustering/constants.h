#pragma once
#include "defines.h"
#include <vector>
#include <string>
#include <nlohmann\json.hpp>

using json = nlohmann::json;

// Weights for MFCCs
constexpr std::array<double, NUM_OF_MFCCS> aWeightsMFCCs{
	{
		2.0,	// MFCC[0]
		1.9,	// MFCC[1]
		1.8,	// MFCC[2]
		1.7,	// MFCC[3]
		1.6,	// MFCC[4]
		1.5,	// MFCC[5]
		1.4,	// MFCC[6]
		1.3,	// MFCC[7]
		1.2,	// MFCC[8]
		1.1,	// MFCC[9]
		1.0,	// MFCC[10]
		0.9,	// MFCC[11]
		0.8		// MFCC[12]
	}
};

// Enum class with all genres
enum class e_Genres
{
	ALTERNATIVE_METAL	= 0,
	BLACK_METAL			= 1,
	DEATH_METAL			= 2,
	CLASSIC_HEAVY_METAL	= 3,
	HARD_ROCK			= 4,
	NU_METAL			= 5,
	THRASH_METAL		= 6,
	PRIMUS				= 7,

	UNDEFINED			= 255,
};

// Songs information
struct s_Song
{
	e_Genres												eGenre;						// Genre of the song
	std::string												strName;					// Name of the song
	std::vector<std::array<double, NUM_OF_MFCCS>>			vecSegments;				// MFCCs of the song
	std::vector<std::vector<double>>						vecFeatures;				// MFCCs + delta + delta-delta
	int														i4Centroid; 				// ID of the centroid assigned to the song
	bool													bWasChanged;				// Flag indicating if the song was assigned to a different centroid
};
// Logging information
struct s_LoggingInfo
{
	int																i4IterationsNum;		// Number of iterations
	bool															bConvergenceAchieved;	// Flag indicating if convergence was achieved
	std::vector<double>												vecPurity;				// Purity of the clusters	
	std::vector<std::vector<double>>								vecInitCentroids;			// Initial centroids	
	std::tm															tStartOfExecution;		// Start time of the execution
};