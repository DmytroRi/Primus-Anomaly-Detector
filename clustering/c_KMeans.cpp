#include <iostream>
#include <fstream>
#include <random>

#include "c_KMeans.h"
#include "constants.h"

c_KMeans::c_KMeans()
:	m_bTerminated		{false}
,	m_i4ClusterNumber	{NUM_OF_CLUSTERS}
{
	m_aMaxMFCC.fill(std::numeric_limits<double>::lowest());
	m_aMinMFCC.fill(std::numeric_limits<double>::infinity());
}

c_KMeans::~c_KMeans()
{

}

void c_KMeans::RunAlgorithm()
{
	m_sLog.tStartOfExecution = GetCurrentTime();
	m_sLog.i4IterationsNum = 0;
	m_sLog.bConvergenceAchieved = false;

	std::cout<<"Start of the K-Means Clustering Algorithm...\n";
	
	if(bReadData())
		std::cout<<"Dataset successfully loaded.\n";
	else
	{
		m_bTerminated = true;
		std::cout<<"Failed to load the dataset. Terminating program.\n";
	}

	NormalizeDataZScore();
	CalculateDeltaAndDeltaDelta();

	if(bInitCentroids() || !m_bTerminated)
		std::cout<<"Initial centroids have been set.\n";
	else
	{
		m_bTerminated = true;
		std::cout<<"Failed to initialize centroids. Terminating program.\n";
	}

	if(m_bTerminated)
		return;

	std::cout<<"\nK-Means execution started.\nIteration number:\n";
	std::cout<<"0\n";
	for (m_sLog.i4IterationsNum; m_sLog.i4IterationsNum < MAX_ITERATIONS; m_sLog.i4IterationsNum++)
	{
		if (!bAssignItems() || m_bTerminated)
		{
			m_bTerminated = true;
			std::cout << "Item assignment failed. Terminating program.\n";
		}

		if (!bCalculateCenters() || m_bTerminated)
		{
			m_bTerminated = true;
			std::cout << "Failed to update centroids. Terminating program.\n";
		}

		if (bIsConvergenceAchieved() || m_bTerminated)
		{
			m_sLog.bConvergenceAchieved = true;
			std::cout << "Convergence achieved.\n";
			break;
		}

		m_sLog.vecPurity.push_back(f8CalculatePurity());

		std::cout << "\033[1A";
		std::cout << "\033[2K";
		std::cout << m_sLog.i4IterationsNum+1 << "\n";
	}

	LogProtocol();

	if(bWriteData() && !m_bTerminated)
		std::cout<<"Clustering results saved successfully.\n";
	else
	{
		m_bTerminated = true;
		std::cout<<"Failed to save clustering results. Terminating program.\n";
	}
}

bool c_KMeans::bReadData()
{
	std::cout<<"Loading data from dataset...\n";

	std::ifstream in{SRC_FILE};
	if (!in.is_open())
	{
		std::cout<<"Failed to open '"<< SRC_FILE<< "'.\n";
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
			sSong.i4Centroid = NUM_OF_CLUSTERS;			// starting dummy value
			sSong.bWasChanged = false;

			for (auto const& [segKey, mfccArr] : segmentsObj.items())
			{
                if (!mfccArr.is_array() || mfccArr.size() != NUM_OF_MFCCS)
				{
					std::cout << "Expected 13-element array for " << genreName << "/" << songName << "/" << segKey;
					return false;
				}

				std::array<double,NUM_OF_MFCCS> mfcc;
				for (int i {0}; i < NUM_OF_MFCCS; i++)
                    mfcc[i] = mfccArr[i].get<double>();

				sSong.vecSegments.push_back(mfcc);
			}

			m_vecDataSet.push_back(std::move(sSong));
		}
	}

	std::cout << "Loaded " << m_vecDataSet.size() << " songs.\n";

	return true;
}

bool c_KMeans::bInitCentroids()
{
	std::random_device rd;
	std::mt19937 gen(rd());

#ifdef D2_SAMPLING	// intialization using k-means++
	// Flatten all segments into one array
	std::vector<std::array<double, NUM_OF_MFCCS>> vecAllSegments{};
	vecAllSegments.reserve(
		std::accumulate(m_vecDataSet.begin(), m_vecDataSet.end(), size_t{0},
        [](size_t acc, auto const& s){ return acc + s.vecSegments.size(); })
	);
	for (auto const& song : m_vecDataSet)
	{
		for (auto const& mfcc : song.vecSegments)
		{
			vecAllSegments.push_back(mfcc);
		}
    }

	// Pick the first centroid randomly
	std::uniform_int_distribution<size_t> pick(0, vecAllSegments.size() - 1);
	m_aCentroids[0] = m_sLog.aInitCentroids[0] = vecAllSegments[pick(gen)];

	std::vector<double> vecMinDist2(vecAllSegments.size());
	for (int i {0}; i < vecAllSegments.size(); i++)
		vecMinDist2[i] = f8CalculateEuclideanDistance(vecAllSegments[i], m_aCentroids[0]);

	for (int k {1}; k < NUM_OF_CLUSTERS; ++k)
	{
        std::discrete_distribution<size_t> dist(vecMinDist2.begin(), vecMinDist2.end());
        size_t idx = dist(gen);
        m_aCentroids[k] = m_sLog.aInitCentroids[k] = vecAllSegments[idx];

        // Update minDist2
        for (size_t i = 0; i < vecAllSegments.size(); ++i)
		{
			double d2 = f8CalculateEuclideanDistance(vecAllSegments[i], m_aCentroids[k]);
			if (d2 < vecMinDist2[i])
				vecMinDist2[i] = d2;
        }
    }
#else	// random initialization
	FindMFCCsBounds();
	for (int c {0}; c < NUM_OF_CLUSTERS; ++c)
	{
		for (int d {0}; d < NUM_OF_MFCCS; ++d)
		{
			std::uniform_real_distribution<double> dist(m_aMinMFCC[d], m_aMaxMFCC[d]);
			m_aCentroids[c][d] = m_sLog.aInitCentroids[c][d] = dist(gen);
		}
	}
#endif	//D2_SAMPLING

	return true;
}

bool c_KMeans::bAssignItems()
{ 
	for (auto & song : m_vecDataSet)
	{
		if (song.vecSegments.empty()) continue;
		std::array<int, NUM_OF_CLUSTERS> aCenroidsCounts{};
		for (auto const & mfcc : song.vecSegments)
		{
			double f8BestDistance{std::numeric_limits<double>::infinity()};
			int i4BestDistPlace{0};
			for (int i{ 0 }; i < NUM_OF_CLUSTERS; i++)
			{
				double f8Distance {f8CalculateEuclideanDistance(mfcc, m_aCentroids[i])};
				if (f8Distance < f8BestDistance)
				{
					f8BestDistance = f8Distance;
					i4BestDistPlace = i;
				}
			}
			aCenroidsCounts[i4BestDistPlace]++;
		}

		// Find the index of the vectors mode 
		auto i4Assign{std::distance(aCenroidsCounts.begin(), 
								   std::max_element(aCenroidsCounts.begin(), aCenroidsCounts.end()))};

		if (song.i4Centroid == i4Assign)
		{
			song.bWasChanged = false;
			continue;
		}
		song.i4Centroid = static_cast<int>(i4Assign);
		song.bWasChanged = true;
	}
	return true;
}

bool c_KMeans::bCalculateCenters()
{
	std::array<std::array<double,NUM_OF_MFCCS>, NUM_OF_CLUSTERS> aSum{};
    std::array<int, NUM_OF_CLUSTERS> aCount{};
    aSum.fill({});
    aCount.fill(0);

	// Collecting values in accumulators
	for (auto const & song : m_vecDataSet)
	{
		int i4CentroidID {song.i4Centroid};
		if(i4CentroidID < 0 || i4CentroidID > NUM_OF_CLUSTERS)
			continue;

		for (auto const & mfcc : song.vecSegments)
		{
			for (int i4MfccID{ 0 }; i4MfccID < NUM_OF_MFCCS; i4MfccID++)
				aSum[i4CentroidID][i4MfccID] += mfcc[i4MfccID];
			aCount[i4CentroidID]++;
		}
	}

	// Finding the mean (new centroids position)
	for (int i4CentroidID{ 0 }; i4CentroidID < NUM_OF_CLUSTERS; i4CentroidID++)
	{
		/*if (aCount[i4CentroidID] == 0)
		{
			// no segments assigned to cluster (reseeding required?)
			return false;
		}*/
		for (int i4MfccID{ 0 }; i4MfccID < NUM_OF_MFCCS; i4MfccID++)
			m_aCentroids[i4CentroidID][i4MfccID] = aSum[i4CentroidID][i4MfccID]/
												   static_cast<double>(aCount[i4CentroidID]);
	}
	return true;
}

bool c_KMeans::bWriteData() const
{
	nlohmann::json j;
    for (auto const& song : m_vecDataSet)
	{
        j[std::to_string(song.i4Centroid)].push_back({
            {"name ",  song.strName},
            {"genre", sEnumGenreToStr(song.eGenre)}
        });
    }
	std::ofstream(RES_FILE) << j.dump(2);
	return true;
}

bool c_KMeans::bIsConvergenceAchieved() const
{
	for (auto const & song : m_vecDataSet)
		if(song.bWasChanged == true)
			return false;

	return true;
}

void c_KMeans::FindMFCCsBounds()
{
	// This function iterates through the data set to find the biggest and the lowest value of each MFCC
	for (auto const & song : m_vecDataSet)
	{
		for (auto const & mfcc : song.vecSegments)
		{
			for (int i{ 0 }; i < NUM_OF_MFCCS; i++)
			{
				m_aMaxMFCC[i] = std::max(m_aMaxMFCC[i], mfcc[i]);
				m_aMinMFCC[i] = std::min(m_aMinMFCC[i], mfcc[i]);
			}
		}
	}
}

void c_KMeans::NormalizeDataZScore()
{
	size_t i4NumOfSegments{ 0 };
	for (auto const & song : m_vecDataSet)
		i4NumOfSegments += song.vecSegments.size();

	std::array<double, NUM_OF_MFCCS> aSum{};
	aSum.fill(0.0);
	std::array<double, NUM_OF_MFCCS> aSumPow2{};
	aSumPow2.fill(0.0);

	for (auto const & song : m_vecDataSet)
		for (auto const & mfcc : song.vecSegments)
			for (int i{ 0 }; i < NUM_OF_MFCCS; i++)
			{
				aSum[i] += mfcc[i];
				aSumPow2[i] += mfcc[i] * mfcc[i];
			}

	std::array<double, NUM_OF_MFCCS> aMean{};
	std::array<double, NUM_OF_MFCCS> aStdDev{};
	for (int i{ 0 }; i < NUM_OF_MFCCS; i++)
	{
		aMean[i] = aSum[i] / static_cast<double>(i4NumOfSegments);
		aStdDev[i] = sqrt((aSumPow2[i] / static_cast<double>(i4NumOfSegments)) - (aMean[i] * aMean[i]));
	}

	// Normalize and save the MFCCs
	for (auto & song : m_vecDataSet)
		for (auto & mfcc : song.vecSegments)
			for (int i{ 0 }; i < NUM_OF_MFCCS; i++)
				if (aStdDev[i] != 0.0)
					mfcc[i] = (mfcc[i] - aMean[i]) / aStdDev[i];
				else
					mfcc[i] = 0.0;

	// Save the normalized MFCCs to the extended features vector
	for (auto & song : m_vecDataSet)
		for (auto & feature : song.vecFeatures)
		{
			for (auto const & mfcc : song.vecSegments)
			{
				feature.push_back(mfcc[0]);
				feature.push_back(mfcc[1]);
				feature.push_back(mfcc[2]);
				feature.push_back(mfcc[3]);
				feature.push_back(mfcc[4]);
				feature.push_back(mfcc[5]);
				feature.push_back(mfcc[6]);
				feature.push_back(mfcc[7]);
				feature.push_back(mfcc[8]);
				feature.push_back(mfcc[9]);
				feature.push_back(mfcc[10]);
				feature.push_back(mfcc[11]);
				feature.push_back(mfcc[12]);
			}
		}

	return;
}

void c_KMeans::CalculateDeltaAndDeltaDelta()
{
	for (auto & song : m_vecDataSet)
	{
		std::vector<std::array<double, NUM_OF_MFCCS>> vecDelta;
		std::vector<std::array<double, NUM_OF_MFCCS>> vecDeltaDelta;

		auto & M {song.vecSegments};
		size_t N {song.vecSegments.size()};

		// Resize delta and delta-delta vectors
		vecDelta.resize(N);
		vecDeltaDelta.resize(N);

		// Compute delta coefficients
		for (int i = 0; i < N; ++i)
		{
			int im1 {i == 0 ? 0 : i - 1};
			int ip1 {i + 1 < N ? i + 1 : i};
            for (int d = 0; d < NUM_OF_MFCCS; ++d)
                vecDelta[i][d] = ( M[ip1][d] - M[im1][d] ) * 0.5;
        }

		// Compute delta-delta coefficients
		for (int i = 0; i < N; ++i)
		{
			int im1{ i == 0 ? 0 : i - 1 };
			int ip1{ i + 1 < N ? i + 1 : i };
			for (int d = 0; d < NUM_OF_MFCCS; ++d)
				vecDeltaDelta[i][d] = (vecDelta[ip1][d] - vecDelta[im1][d]) * 0.5;
		}

		// Append delta and delta-delta coefficients to the extended MFCCs
		for (auto & segment : song.vecSegmentsExtended)
		{
			for (int d{ 0 }; d < NUM_OF_MFCCS; d++)
			{
				segment[d + NUM_OF_MFCCS] = vecDelta[&segment - &song.vecSegmentsExtended[0]][d];
				segment[d + 2 * NUM_OF_MFCCS] = vecDeltaDelta[&segment - &song.vecSegmentsExtended[0]][d];
			}
		}

	}
	return;
}

void c_KMeans::LogProtocol()
{
	std::tm end {GetCurrentTime()};

	std::ofstream out{LOG_FILE, std::ios::app};
	if (!out)
	{
        std::cout << "Error: could not open " << LOG_FILE << " for logging\n";
        return;
    }

	out<<"\n=== K-Means Clustering Algorithm ===\n";
	out<<"Execution started at:\t"<< std::put_time(&m_sLog.tStartOfExecution, "%Y-%m-%d %H:%M:%S") << "\n";
	out<<"Execution ended at:\t\t"<< std::put_time(&end, "%Y-%m-%d %H:%M:%S") << "\n";
	out<<"Initial centroids:\n";
	for (int c{ 0 }; c < NUM_OF_CLUSTERS; c++)
	{
		out<<"Centoroid #"<<c<<": [";
		for (int d{ 0 }; d < NUM_OF_MFCCS; d++)
		{
			out << std::setw(10) << std::fixed << std::setprecision(4) << m_sLog.aInitCentroids[c][d];
			if (d < NUM_OF_MFCCS - 1)
				out << ", ";
		}
			
		out<<"]\n";
	}
	out << "Amount of executed iterations: " << m_sLog.i4IterationsNum << "\n";
	out << "Convergence achieved: " << (m_sLog.bConvergenceAchieved ? "Yes" : "No") << "\n";

	for (int i{ 0 }; i < m_sLog.vecPurity.size(); i++)
		out << "Purity after iteration #" << i+1 << ": " << std::fixed << std::setprecision(4) << m_sLog.vecPurity[i] << "\n";

#ifdef EXTENDED_LOGGING
	std::string sEntry{};
	for (auto const & song : m_vecDataSet)
	{
		sEntry = sEnumGenreToStr(song.eGenre);
		out << sEntry << "/" << song.strName << ":\t";
		out << "Assigned to cluster #" << song.i4Centroid << ", ";
		out << "Was changed: " << (song.bWasChanged ? "Yes" : "No") << "\n";
	}
#endif
	out << "====================================\n";
	out.close();
	std::cout << "Log saved to " << LOG_FILE << ".\n";
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
	else if(sGenre == "Primus" || sGenre == "primus")
		return e_Genres::PRIMUS;
	else
		return e_Genres::UNDEFINED;
}

double c_KMeans::f8CalculateEuclideanDistance(const std::array<double, NUM_OF_MFCCS> & a, const std::array<double, NUM_OF_MFCCS> & b, const bool isSqrt /*=false*/) const
{
	double f8Sum{0};
	for (int i{ 0 }; i < NUM_OF_MFCCS; i++)
	{
		double dist {a[i]-b[i]};
		f8Sum += dist*dist;
	}
	if(isSqrt)
		return sqrt(f8Sum);
	return f8Sum;
}

double c_KMeans::f8CalculatePurity() const
{
	std::array<std::array<size_t,NUM_OF_CLUSTERS>,NUM_OF_CLUSTERS> aCounts{};
	for (auto & row : aCounts) row.fill(0);

	for (auto const& song : m_vecDataSet)
	{
        int k = song.i4Centroid;
        int g = static_cast<int>(song.eGenre);
        if (k >= 0 && k < NUM_OF_CLUSTERS
			&& g >= 0 && g < NUM_OF_CLUSTERS)
            aCounts[k][g]++;
    }

	size_t i4Total{ 0 };
	for (int i{ 0 }; i < NUM_OF_CLUSTERS; i++)
	{
		size_t i4MaxCount {*std::max_element(aCounts[i].begin(), aCounts[i].end())};
		i4Total += i4MaxCount;
	}

	return static_cast<double>(i4Total) / m_vecDataSet.size();
}

std::tm c_KMeans::GetCurrentTime() const
{
	auto now   = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);	// get current time
    std::tm now_tm;
    localtime_s(&now_tm, &now_t);							// convert to local time zone
	return now_tm;
}
