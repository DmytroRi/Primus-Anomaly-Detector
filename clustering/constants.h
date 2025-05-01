#pragma once
#include <vector>
#include <string>
#include <nlohmann\json.hpp>


using json = nlohmann::json;

#define NUM_OF_MFCCS  13				// Amount of MFCCs in dataset
#define NUM_OF_CLUSTERS  8				// Amount of clusters (note: must be either n or n-1, where n is amount of genres) 
#define D2_SAMPLING						// Use k-means++ initialization for initial centroids
#define MAX_ITERATIONS 500'000			// Max number of runs
#define SRC_FILE "data_mean_15s.json"	// Name of the dataset file
#define RES_FILE "RESULT.JSON"			// Name of the result file
#define LOG_FILE "Protocol.txt"			// Name of the logging file
//#define EXTENDED_LOGGING				// Enable extended logging

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
	std::vector<std::array<double, NUM_OF_MFCCS * 3>>		vecSegmentsExtended;		// MFCCs + delta + delta-delta
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
	std::array<std::array<double, NUM_OF_MFCCS>, NUM_OF_CLUSTERS>	aInitCentroids;			// Initial centroids	
	std::tm															tStartOfExecution;		// Start time of the execution
};