#include <iostream>
#include <fstream>
#include <nlohmann\json.hpp>

#include "c_KMeans.h"
#include "constants.h"

c_KMeans::c_KMeans()
:	m_bTerminated		{false}
,	m_i4ClusterNumber	{NUM_OF_CLUSTERS}
{
	
}

c_KMeans::~c_KMeans()
{

}

void c_KMeans::RunAlgorithm()
{
	std::cout<<"Start of the K-Means Clustering Algorithm.\n";
	
	if(bReadData())
		std::cout<<"The dataset was succesfully loaded.\n";
	else
	{
		m_bTerminated = true;
		std::cout<<"The dataset upload was failed. Termination of program.\n";
	}

}

bool c_KMeans::bReadData()
{
	std::cout<<"Loading data from the dataset.\n";

	std::ifstream in{"data_mean_15s.json"};
	if (!in.is_open())
	{
		std::cout<<"Could not open data_mean_15s.json\n";
		return false;
	}

    json j; 
    in >> j;

	for (auto const & [genreName, songsObj] : j.items())
	{
		for (auto const & [songName, segmentsObj] : songsObj.items())
		{
			s_Song sSong;
			sSong.strName = songName;

			for (auto const& [segKey, mfccArr] : segmentsObj.items())
			{
                if (!mfccArr.is_array() || mfccArr.size() != 13)
				{
					std::cout << "Expected 13-element array for " << genreName << "/" << songName << "/" << segKey;
					return false;
				}

				std::array<double,NUM_OF_MFCCS> mfcc;
				for (size_t i {0}; i < NUM_OF_MFCCS; i++)
                    mfcc[i] = mfccArr[i].get<double>();

				sSong.vecSegments.push_back(mfcc);
			}

			m_vecDataSet.push_back(std::move(sSong));
		}
	}

	std::cout << "Loaded " << m_vecDataSet.size() << " songs\n";

	return true;
}

bool c_KMeans::bInitCentroids()
{
	return false;
}

bool c_KMeans::bAssignItems()
{
	return false;
}

bool c_KMeans::bCalculateCenters()
{
	return false;
}

bool c_KMeans::bWriteData()
{
	return false;
}
