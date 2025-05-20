#include <iostream>
#include <fstream>
#include <random>

#include "c_KMeans.h"
#include "constants.h"

c_AlgorithmBase::c_AlgorithmBase()
:	m_bTerminated		{false}
,	m_i4ClusterNumber	{NUM_OF_CLUSTERS}
{}

bool c_AlgorithmBase::bReadData()
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

			auto itFramesStart = segmentsObj.find("frames");
			if (itFramesStart == segmentsObj.end() || !itFramesStart->is_array())
			{
				std::cout << "Missing or invalid 'frames' array for " << genreName << "/" << songName << "\n";
				return false;
			}

			for (auto const& features : *itFramesStart)
			{
                if (!features.is_array() || features.size() != NUM_OF_FEATURES)
				{
					std::cout << "Expected " << NUM_OF_FEATURES << "-element array for " << genreName << "/" << songName << "\n";
					return false;
				}

				std::array<double,NUM_OF_FEATURES> aFeaturesToSave;
				for (int i {0}; i < NUM_OF_FEATURES; i++)
                    aFeaturesToSave[i] = features[i].get<double>();

				sSong.vecSegments.push_back(aFeaturesToSave);
			}

			m_vecDataSet.push_back(std::move(sSong));
		}
	}

	std::cout << "Loaded " << m_vecDataSet.size() << " songs.\n";

	in.close();
	return true;
}

bool c_AlgorithmBase::bWriteData() const
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

double c_AlgorithmBase::f8CalculateEuclideanDistance(const std::vector<double> & a, const std::vector<double> & b, const bool isSqrt /*=false*/) const
{
	double f8Sum{0};
	for (int i{ 0 }; i < NUM_OF_FEATURES; i++)
	{
		double dist {a[i]-b[i]};
#ifdef WEIGHTED_MFCCS
		i < NUM_OF_MFCCS ? f8Sum += dist * dist * aWeightsMFCCs[i]
					     : f8Sum += dist * dist;
#else
		f8Sum += dist * dist;
#endif // WEIGHTED_MFCCS
	}
	if(isSqrt)
		return sqrt(f8Sum);
	return f8Sum;
}

void c_AlgorithmBase::NormalizeDataZScore()
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
			for (auto const & mfcc : song.vecSegments)
			{
				song.vecFeatures.push_back({mfcc[0], mfcc[1], mfcc[2],mfcc[3],
											mfcc[4], mfcc[5], mfcc[6],mfcc[7],
											mfcc[8], mfcc[9], mfcc[10], mfcc[11], mfcc[12],});
			}

	return;
}

void c_AlgorithmBase::CalculateDeltaAndDeltaDelta()
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

		// Append delta and delta-delta coefficients to the features
		for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j < NUM_OF_MFCCS; ++j)
				song.vecFeatures[i].push_back(vecDelta[i][j]);
			for (int j = 0; j < NUM_OF_MFCCS; ++j)
				song.vecFeatures[i].push_back(vecDeltaDelta[i][j]);
		}

	}
	return;
}

std::tm c_AlgorithmBase::GetCurrentTime() const
{
	auto now   = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);	// get current time
    std::tm now_tm;
    localtime_s(&now_tm, &now_t);							// convert to local time zone
	return now_tm;
}

std::string c_AlgorithmBase::sEnumGenreToStr(const e_Genres & eGenre) const
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

e_Genres c_AlgorithmBase::eStrGenreToEnum(const std::string & sGenre) const
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

c_KMeans::c_KMeans()
:	c_AlgorithmBase()
{
	m_aMaxMFCC.fill(std::numeric_limits<double>::lowest());
	m_aMinMFCC.fill(std::numeric_limits<double>::infinity());
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



bool c_KMeans::bInitCentroids()
{
	std::random_device rd;
	std::mt19937 gen(rd());

#ifdef D2_SAMPLING	// intialization using k-means++
	// Flatten all segments into one array
	std::vector<std::vector<double>> vecAllSegments{};
	//vecAllSegments.reserve(
	//	std::accumulate(m_vecDataSet.begin(), m_vecDataSet.end(), size_t{0},
    //    [](size_t acc, auto const& s){ return acc + s.vecSegments.size(); })
	//);
	for (auto const& song : m_vecDataSet)
	{
		for (auto const& features : song.vecFeatures)
		{
			vecAllSegments.push_back(features);
		}
    }

	// Pick the first centroid randomly
	std::uniform_int_distribution<size_t> pick(0, vecAllSegments.size() - 1);
	m_vecCentroids.push_back(vecAllSegments[pick(gen)]);
	m_sLog.vecInitCentroids.push_back(m_vecCentroids[0]);

	std::vector<double> vecMinDist2(vecAllSegments.size());
	for (int i {0}; i < vecAllSegments.size(); i++)
		vecMinDist2[i] = f8CalculateEuclideanDistance(vecAllSegments[i], m_vecCentroids[0]);

	for (int k {1}; k < NUM_OF_CLUSTERS; k++)
	{
        std::discrete_distribution<size_t> dist(vecMinDist2.begin(), vecMinDist2.end());
        size_t idx = dist(gen);
        m_vecCentroids.push_back(vecAllSegments[idx]);
		m_sLog.vecInitCentroids.push_back(vecAllSegments[idx]);

        // Update minDist2
        for (size_t i = 0; i < vecAllSegments.size(); i++)
		{
			double d2 = f8CalculateEuclideanDistance(vecAllSegments[i], m_vecCentroids[k]);
			if (d2 < vecMinDist2[i])
				vecMinDist2[i] = d2;
        }
    }
#else	// random initialization
	FindMFCCsBounds();
	m_vecCentroids.resize(NUM_OF_CLUSTERS);
	m_sLog.vecInitCentroids.resize(NUM_OF_CLUSTERS);
	for (int c {0}; c < NUM_OF_CLUSTERS; ++c)
	{
		m_vecCentroids[c].resize(NUM_OF_FEATURES);
		m_sLog.vecInitCentroids[c].resize(NUM_OF_FEATURES);
		for (int d {0}; d < NUM_OF_MFCCS; ++d)
		{
			// Generate random value between min and max
			std::uniform_real_distribution<double> dist(m_aMinMFCC[d], m_aMaxMFCC[d]);
			// Assign random value to centroid
			m_vecCentroids[c][d] = m_sLog.vecInitCentroids[c][d] = dist(gen);
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
		std::array<int, NUM_OF_FEATURES> aCenroidsCounts{};
		for (auto const & features : song.vecFeatures)
		{
			double f8BestDistance{std::numeric_limits<double>::infinity()};
			int i4BestDistPlace{0};
			for (int i{ 0 }; i < NUM_OF_CLUSTERS; i++)
			{
				double f8Distance {f8CalculateEuclideanDistance(features, m_vecCentroids[i])};
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
	std::array<std::array<double,NUM_OF_FEATURES>, NUM_OF_CLUSTERS> aSum{};
    std::array<int, NUM_OF_CLUSTERS> aCount{};
    aSum.fill({});
    aCount.fill(0);

	// Collecting values in accumulators
	for (auto const & song : m_vecDataSet)
	{
		int i4CentroidID {song.i4Centroid};
		if(i4CentroidID < 0 || i4CentroidID > NUM_OF_CLUSTERS)
			continue;

		for (auto const & features : song.vecFeatures)
		{
			for (int i4MfccID{ 0 }; i4MfccID < NUM_OF_FEATURES; i4MfccID++)
				aSum[i4CentroidID][i4MfccID] += features[i4MfccID];
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
		for (int i4MfccID{ 0 }; i4MfccID < NUM_OF_FEATURES; i4MfccID++)
			m_vecCentroids[i4CentroidID][i4MfccID] = aSum[i4CentroidID][i4MfccID]/
												   static_cast<double>(aCount[i4CentroidID]);
	}
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
	out<<"Source file:\t\t\t" << SRC_FILE << "\n";
	out<<"Execution started at:\t"<< std::put_time(&m_sLog.tStartOfExecution, "%Y-%m-%d %H:%M:%S") << "\n";
	out<<"Execution ended at:\t\t"<< std::put_time(&end, "%Y-%m-%d %H:%M:%S") << "\n";
	out<<"Initial centroids:\n";
	for (int c{ 0 }; c < NUM_OF_CLUSTERS; c++)
	{
		out<<"Centoroid #"<<c<<": [";
		for (int d{ 0 }; d < NUM_OF_FEATURES; d++)
		{
			out << std::setw(10) << std::fixed << std::setprecision(4) << m_sLog.vecInitCentroids[c][d];
			if (d < NUM_OF_FEATURES - 1)
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

c_KNN::c_KNN()
:	c_AlgorithmBase()
,	m_f8TrainRatio{ TRAIN_RATIO }
{
}

void c_KNN::RunAlgorithm()
{
	m_sLog.tStartOfExecution = GetCurrentTime();
	std::cout<<"Start of the k-nearest neighbors Algorithm...\n";
	
	if(bReadData())
		std::cout<<"Dataset successfully loaded.\n";
	else
	{
		m_bTerminated = true;
		std::cout<<"Failed to load the dataset. Terminating program.\n";
	}

	// Preapare the features
	NormalizeDataZScore();
	CalculateDeltaAndDeltaDelta();

	// Split the dataset into training and testing sets
	if(splitDataSet())
		std::cout << "Dataset successfully split into training and testing sets.\n";
	else
	{
		m_bTerminated = true;
		std::cout << "Failed to split the dataset. Terminating program.\n";
	}

	if (m_bTerminated)
		return;

	// Predict the genres of the songs in the training set
	predictAll();
	std::cout << "Predictions completed.\n";

	// Calculate the purity of the training set
	m_sLog.vecPurity.push_back(f8CalculatePurity());

	// Log the results
	LogProtocol();

	return;
}

bool c_KNN::splitDataSet()
{
	std::cout << "Splitting dataset into training and testing sets...\n";
	std::vector<s_Song> vecShuffled{m_vecDataSet};

	std::random_device rd;
	std::mt19937 gen(rd());
	std::shuffle(vecShuffled.begin(), vecShuffled.end(), gen);

	size_t i4TrainSize = static_cast<size_t>(vecShuffled.size() * m_f8TrainRatio);

	m_vecTrainSet.assign(vecShuffled.begin(), vecShuffled.begin() + i4TrainSize);
	m_vecTestSet.assign(vecShuffled.begin() + i4TrainSize, vecShuffled.end());

	if (m_vecTrainSet.empty() || m_vecTestSet.empty())
		return false;

	return true;
}

void c_KNN::optimizeValueK(int i4MaxK, int i4MinK, int i4Step)
{
	if (i4MaxK < i4MinK || i4Step <= 0)
	{
		std::cout << "Invalid range for k. Please check the values.\n";
		return;
	}

	bReadData();
	NormalizeDataZScore();
	CalculateDeltaAndDeltaDelta();
	splitDataSet();
	
	std::cout << "Optimizing the value of k...\n";
	for (int i4K {i4MinK}; i4K <= i4MaxK; i4K += i4Step)
	{
		predictAll(i4K);
		m_sLog.vecPurity.push_back(f8CalculatePurity());
	}

	std::cout << "Purity values for different k:\n";
	for (int i4K{ i4MinK }; i4K <= i4MaxK; i4K += i4Step)
	{
		std::cout << "k = " << i4K << ": " << std::fixed << std::setprecision(4) << m_sLog.vecPurity[i4K - i4MinK] << "\n";
	}

	LogResearchResults();
	return;
}

void c_KNN::predictAll(int i4Neighboor/*=0*/)
{
	std::cout << "Predicting genres for the training set...\n";

	for (auto const& song : m_vecTestSet) {
        e_Genres pred = predict(song, i4Neighboor);
        m_vecPredictions.push_back(pred);
    }
}

e_Genres c_KNN::predict(const s_Song & song, int i4Neighboor)
{
	// Calculate the mean of the requested song
	std::vector<double> vecSongMean(NUM_OF_FEATURES, 0.0);
	for (auto const & seg : song.vecFeatures)
		for (size_t d = 0; d < NUM_OF_FEATURES; ++d)
			vecSongMean[d] += seg[d];

	double invQ = 1.0 / static_cast<double>(song.vecFeatures.size());
	for (size_t d = 0; d < NUM_OF_FEATURES; ++d) vecSongMean[d] *= invQ;

	// Calculate distances to all training songs
	std::vector<std::pair<double, e_Genres>> distances;
	for (auto const & trainSong : m_vecTrainSet)
	{
		// Calculate the mean of the training song
		std::vector<double> vecTrainSongMean(NUM_OF_FEATURES, 0.0);
		for (auto const & seg : trainSong.vecFeatures)
			for (size_t d = 0; d < NUM_OF_FEATURES; ++d)
				vecTrainSongMean[d] += seg[d];

		double invQ = 1.0 / static_cast<double>(trainSong.vecFeatures.size());
		for (size_t d = 0; d < NUM_OF_FEATURES; ++d) vecTrainSongMean[d] *= invQ;

		// Calculate the distance
		double distance = f8CalculateEuclideanDistance(vecSongMean, vecTrainSongMean);
		distances.emplace_back(distance, trainSong.eGenre);
	}

	// Choose the number of neighbors to consider
	int i4NeighboorCount{NEIGHBOUR_COUNT};
	if (i4Neighboor != NEIGHBOR_COUNT_DUMMY)
		i4NeighboorCount = i4Neighboor;


	// Sort distances
	if (i4NeighboorCount < distances.size())
	{
		std::nth_element(
			distances.begin(),
			distances.begin() + i4NeighboorCount,
			distances.end(),
			[](auto & a, auto & b) { return a.first < b.first; }
		);
	}

	std::array<size_t, NUM_OF_CLUSTERS> votes{};
	votes.fill(0);
	size_t limit = std::min(static_cast<size_t>(i4NeighboorCount), distances.size());
	for (size_t i = 0; i < limit; ++i)
		votes[static_cast<int>(distances[i].second)]++;

	// Find the genre with the most votes
	long long i8BestIdx = std::distance(
		votes.begin(),
		std::max_element(votes.begin(), votes.end())
	);

	return static_cast<e_Genres>(i8BestIdx);
}

void c_KNN::LogProtocol()
{
	std::tm end{ GetCurrentTime() };

	std::ofstream out{ LOG_FILE, std::ios::app };
	if (!out)
	{
		std::cout << "Error: could not open " << LOG_FILE << " for logging\n";
		return;
	}

	out << "\n=== k-Nearest Neighbors Algorithm ===\n";
	out << "Source file:\t\t\t" << SRC_FILE << "\n";
	out << "Execution started at:\t" << std::put_time(&m_sLog.tStartOfExecution, "%Y-%m-%d %H:%M:%S") << "\n";
	out << "Execution ended at:\t\t" << std::put_time(&end, "%Y-%m-%d %H:%M:%S") << "\n";
	out << "Value of k:\t\t\t\t" << NEIGHBOUR_COUNT << "\n";
	out << "Achieved purity:\t\t" << std::fixed << std::setprecision(4) << m_sLog.vecPurity[0] << "\n";
	out << "====================================\n";

	out.close();
	std::cout << "Log saved to " << LOG_FILE << ".\n";
}

double c_KNN::f8CalculatePurity()
{
	if (m_vecTrainSet.empty())
		return 0.0;

	int i4CorrectPredictions{ 0 };
	for (size_t i{ 0 }; i < m_vecTestSet.size(); i++)
	{
		if (m_vecTestSet[i].eGenre == m_vecPredictions[i])
			i4CorrectPredictions++;
	}

	// Clear the predictions vector
    m_vecPredictions.clear();

	return static_cast<double>(i4CorrectPredictions)/m_vecTestSet.size();
}

void c_KNN::LogResearchResults(int i4MaxK, int i4MinK, int i4Step)
{
	std::ofstream out{LOG_RESEARCH, std::ios::app};
	if (!out)
	{
        std::cout << "Error: could not open " << LOG_RESEARCH << " for logging\n";
        return;
    }

	out << "\n=== k-Nearest Neighbors Algorithm ===\n";
	out << "Source file:\t\t\t" << SRC_FILE << "\n";
	out << "Number of features:\t\t" << NUM_OF_FEATURES << "\n";
	out << "Number of MFCCs:\t\t" << NUM_OF_MFCCS << "\n";
	out << "Number of clusters:\t\t" << NUM_OF_CLUSTERS << "\n";
	out << "Train ratio:\t\t\t" << m_f8TrainRatio << "\n";
    #ifdef WEIGHTED_MFCCS
    out << "Weighted MFCCs:\t\t\tYes\n";
    #else
    out << "Weighted MFCCs:\t\t\tNo\n";
    #endif
	out << "Results:\nValues of k:\t\t";
	for (int i4K = i4MinK; i4K <= i4MaxK; i4K += i4Step)
	{
		out << std::setw(8) << i4K;
	}
	out << "\n";
	out << "Purity values: \t\t\t";
	for (int i4K = i4MinK; i4K <= i4MaxK; i4K += i4Step)
	{
		out << std::setw(8) << std::fixed << std::setprecision(4) << m_sLog.vecPurity[i4K - i4MinK];
	}
	out << "\n";
	out << "====================================\n";
	out.close();
	std::cout << "Research results saved to " << LOG_RESEARCH << ".\n";
	return;
}
