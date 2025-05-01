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
	e_Genres												eGenre;
	std::string												strName;
	std::vector<std::array<double,NUM_OF_MFCCS>>			vecSegments;
	std::vector<std::array<double, NUM_OF_MFCCS * 3>>		vecSegmentsExtended;	
	int														i4Centroid;
	bool													bWasChanged;
};
// Logging information
struct s_LoggingInfo
{
	int																i4IterationsNum;
	bool															bConvergenceAchieved;
	std::vector<double>												vecPurity;	
	std::array<std::array<double, NUM_OF_MFCCS>, NUM_OF_CLUSTERS>	aInitCentroids;
	std::tm															tStartOfExecution;
};