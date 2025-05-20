#pragma once

// Constants for the program
#define NUM_OF_FEATURES  13				// Amount of features in dataset per song ( 13 MFCCs + 13 delta + 13 delta-delta)
#define NUM_OF_MFCCS  13				// Amount of MFCCs in dataset
#define NUM_OF_CLUSTERS  8				// Amount of clusters (note: must be either n or n-1, where n is amount of genres) 
#define D2_SAMPLING						// Use k-means++ initialization for initial centroids
#define MAX_ITERATIONS 500'000			// Max number of runs

//#define EXTENDED_LOGGING				// Enable extended logging
#define WEIGHTED_MFCCS					// Enable weighted MFCCs
#define TRAIN_RATIO 0.8					// Ratio of training data to total data
#define NEIGHBOUR_COUNT 15				// Number of neighbors for KNN algorithm
#define NEIGHBOR_COUNT_DUMMY -1			// Dummy value for KNN algorithm

// File names
#define RES_FILE "RESULT.JSON"											// Name of the result file
#define LOG_FILE "Protocol.txt"											// Name of the logging file
#define LOG_RESEARCH "Research.txt"										// Name of the research logging file
#define SRC_FILE "../computed_data/features_13mfcc_20frame_10hop_nodeltas_nocmvn.json"	// Name of the dataset file
