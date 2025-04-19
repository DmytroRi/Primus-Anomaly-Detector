#include <iostream>
#include <fstream>

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

	m_vecDataSet;
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
			sSong.eGenre = eStrGenreToEnum(genreName);
			sSong.strName = songName;

			for (auto const& [segKey, mfccArr] : segmentsObj.items())
			{
                if (!mfccArr.is_array() || mfccArr.size() != NUM_OF_MFCCS)
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

std::string c_KMeans::sEnumGenreToStr(const e_Genres & eGenre) const
{
	switch (eGenre)
	{
	case e_Genres::ALTERNATIVE_METAL:	return "Alternative metal";
	case e_Genres::BLACK_METAL:			return "Black metal";
	case e_Genres::DEATH_METAL:			return "Death metal";
	case e_Genres::CLASSIC_HEAVY_METAL:	return "Classic heavy metal";
	case e_Genres::HARD_ROCK:			return "Hard rock";
	case e_Genres::NU_METAL:			return "Nu metal";
	case e_Genres::THRASH_METAL:		return "Thrash metal";
	case e_Genres::PRIMUS:				return "Primus";

	default:							return "Undefined";
	}
}

e_Genres c_KMeans::eStrGenreToEnum(const std::string & sGenre) const
{
	if(sGenre == "Alternative metal" || sGenre == "alternative_metal")
		return e_Genres::ALTERNATIVE_METAL;
	else if(sGenre == "Black metal" || sGenre == "black_metal")
		return e_Genres::BLACK_METAL;
	else if(sGenre == "Death metal" || sGenre == "death_metal")
		return e_Genres::DEATH_METAL;
	else if(sGenre == "Classic heavy metal" || sGenre == "classic_heavy_metal")
		return e_Genres::CLASSIC_HEAVY_METAL;
	else if(sGenre == "Hard rock" || sGenre == "hard_rock")
		return e_Genres::HARD_ROCK;
	else if(sGenre == "Nu metal" || sGenre == "nu_metal")
		return e_Genres::NU_METAL;
	else if(sGenre == "Thrash metal" || sGenre == "thrash_metal")
		return e_Genres::THRASH_METAL;
	else if(sGenre == "Primus metal" || sGenre == "primus_metal")
		return e_Genres::PRIMUS;
	else
		return e_Genres::UNDEFINED;
}
