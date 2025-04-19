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
	bool			bWriteData();

	std::string		sEnumGenreToStr(const e_Genres & eGenre) const;
	e_Genres		eStrGenreToEnum(const std::string & sGenre) const;
	

	bool					m_bTerminated{};
	int						m_i4ClusterNumber{};
	std::vector<s_Song>		m_vecDataSet{};
};

