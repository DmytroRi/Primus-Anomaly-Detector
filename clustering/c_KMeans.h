#pragma once
#include "constants.h"

// This is the base class for all algorithms
class c_AlgorithmBase
{
public:
	c_AlgorithmBase();
	virtual				~c_AlgorithmBase() = default;
	virtual void		RunAlgorithm() = 0;

protected:
	// Data loading and saving
	bool				bReadData();
	bool				bWriteData() const;
	virtual void		LogProtocol() = 0;

	// Feature extraction and processing
	void				NormalizeDataZScore();
	void				CalculateDeltaAndDeltaDelta();
	double				f8CalculateEuclideanDistance(const std::vector<double>& a,
													 const std::vector<double>& b,
													 const bool isSqrt = false) const;

	// Logging and assisting functions
	std::tm				GetCurrentTime() const;
	std::string			sEnumGenreToStr(const e_Genres & eGenre) const;
	e_Genres			eStrGenreToEnum(const std::string & sGenre) const;
	virtual double		f8CalculatePurity() const = 0;

	// Data members
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
	~c_KMeans() = default;

	void			RunAlgorithm();

private:
	// Initialization functions
	bool			bInitCentroids();

	// Workflow functions
	bool			bAssignItems();
	bool			bCalculateCenters();
	bool			bIsConvergenceAchieved() const;

	// Logging functions
	void			LogProtocol();

	// Helper functions
	void			FindMFCCsBounds();
	double			f8CalculatePurity() const;
	
	// Data members
	std::array<double, NUM_OF_MFCCS>	m_aMaxMFCC{};
	std::array<double, NUM_OF_MFCCS>	m_aMinMFCC{};
	std::vector<std::vector<double>>	m_vecCentroids{};
};

class c_KNN : public c_AlgorithmBase
{
public:
	c_KNN();
	~c_KNN() = default;

	void			RunAlgorithm();

private:
	bool			splitDataSet();
	void			predictAll();
	e_Genres		predict(const s_Song & song);

	void			LogProtocol();

	double			f8CalculatePurity() const;


	double								m_f8TrainRatio{};
	std::vector<s_Song>					m_vecTrainSet{};
	std::vector<s_Song>					m_vecTestSet{};
	std::vector<e_Genres>				m_vecPredictions{};
};