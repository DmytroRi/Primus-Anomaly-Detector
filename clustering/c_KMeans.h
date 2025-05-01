#pragma once
#include "constants.h"

// Implementation of K-means clustering from scratch 
class c_KMeans
{
public:
	c_KMeans();
	~c_KMeans();

	void			RunAlgorithm();

private:
	bool			bReadData();
	bool			bInitCentroids();
	bool			bAssignItems();
	bool			bCalculateCenters();
	bool			bWriteData() const;

	bool			bIsConvergenceAchieved() const;

	void			FindMFCCsBounds();
	void			NormalizeDataZScore();
	void			CalculateDeltaAndDeltaDelta();

	void			LogProtocol();

	std::string		sEnumGenreToStr(const e_Genres & eGenre) const;
	e_Genres		eStrGenreToEnum(const std::string & sGenre) const;

	double			f8CalculateEuclideanDistance(const std::vector<double>& a,
												 const std::vector<double>& b,
												 const bool isSqrt = false) const;
	double			f8CalculatePurity() const;
	
	std::tm	GetCurrentTime() const;

	bool								m_bTerminated{};
	int									m_i4ClusterNumber{};
	std::vector<s_Song>					m_vecDataSet{};
	std::array<double, NUM_OF_MFCCS>	m_aMaxMFCC{};
	std::array<double, NUM_OF_MFCCS>	m_aMinMFCC{};
	s_LoggingInfo						m_sLog{};


	std::vector<std::vector<double>>	m_vecCentroids{};
};

