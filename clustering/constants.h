#pragma once
#include <vector>
#include <string>
#include <nlohmann\json.hpp>


using json = nlohmann::json;

#define NUM_OF_FEATURES  39				// Amount of features in dataset per song ( 13 MFCCs + 13 delta + 13 delta-delta)
#define NUM_OF_MFCCS  13				// Amount of MFCCs in dataset
#define NUM_OF_CLUSTERS  8				// Amount of clusters (note: must be either n or n-1, where n is amount of genres) 
#define D2_SAMPLING						// Use k-means++ initialization for initial centroids
#define MAX_ITERATIONS 500'000			// Max number of runs
#define SRC_FILE "data_mean_1s.json"	// Name of the dataset file
#define RES_FILE "RESULT.JSON"			// Name of the result file
#define LOG_FILE "Protocol.txt"			// Name of the logging file
//#define EXTENDED_LOGGING				// Enable extended logging
#define WEIGHTED_MFCCS					// Enable weighted MFCCs
#define TRAIN_RATIO 0.8					// Ratio of training data to total data
#define NEIGHBOUR_COUNT 15				// Number of neighbors for KNN algorithm

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