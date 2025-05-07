#pragma once
#include "constants.h"

// This is the base class for all algorithms
class c_AlgorithmBase
{
public:
	c_AlgorithmBase();
	virtual ~c_AlgorithmBase() = default;
	virtual void RunAlgorithm() = 0;

protected:
	virtual void	LogProtocol() = 0;
	bool			bReadData();
	bool			bWriteData() const;

	double				f8CalculateEuclideanDistance(const std::vector<double>& a,
													 const std::vector<double>& b,
													 const bool isSqrt = false) const;

	void			NormalizeDataZScore();
	void			CalculateDeltaAndDeltaDelta();

	std::tm				GetCurrentTime() const;
	std::string			sEnumGenreToStr(const e_Genres & eGenre) const;
	e_Genres			eStrGenreToEnum(const std::string & sGenre) const;


	bool								m_bTerminated{};
	int									m_i4ClusterNumber{};
	std::vector<s_Song>					m_vecDataSet{};
	s_LoggingInfo						m_sLog{};
};

// Implementation of K-means clustering from scratch 
class c_KMeans : public c_AlgorithmBase
{
public:
	c_KMeans();
	~c_KMeans();

	void			RunAlgorithm();

private:
	//bool			bReadData();
	bool			bInitCentroids();
	bool			bAssignItems();
	bool			bCalculateCenters();

	bool			bIsConvergenceAchieved() const;

	void			FindMFCCsBounds();
	

	void			LogProtocol();

	double			f8CalculatePurity() const;
	

	//bool								m_bTerminated{};
	//int									m_i4ClusterNumber{};
	//std::vector<s_Song>					m_vecDataSet{};
	std::array<double, NUM_OF_MFCCS>	m_aMaxMFCC{};
	std::array<double, NUM_OF_MFCCS>	m_aMinMFCC{};
	//s_LoggingInfo						m_sLog{};


	std::vector<std::vector<double>>	m_vecCentroids{};
};

