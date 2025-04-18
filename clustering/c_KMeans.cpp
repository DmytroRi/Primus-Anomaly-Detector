#include <iostream>

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
		std::cout<<"The dataset was succesfully uploaded.\n";
	else
	{
		m_bTerminated = true;
		std::cout<<"The dataset upload was failed. Termination of program.\n";
	}

}

bool c_KMeans::bReadData()
{
	std::cout<<"Uploading data from the dataset.\n";
	return false;
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
