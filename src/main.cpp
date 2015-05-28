
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "net.h"

double frand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main() {
	srand(time(NULL));
	std::vector<unsigned int> topology;
	topology.push_back(2);
	topology.push_back(1);
	topology.push_back(5);
	topology.push_back(5);
	topology.push_back(1);
	Net nn(topology);

	int i = 0;
	while (i < 10000) {
		std::vector<double> inputVals;
		double x = frand(0.2, 0.5);
		double y = frand(0.2, 0.5);

		inputVals = {(double)x, (double)y};
		nn.feedForward(inputVals);
		std::vector<double> targetVals;
		targetVals = {x + y};
		
		std::vector<double> resultVals;

		nn.getResults(resultVals);
		
		for (auto c : resultVals) {
			std::cout << x << " " << y << " should be " <<(x + y) << " actual is " << c <<  std::endl;
		}
		    // Report how well the training is working, average over recent samples:
        std::cout << "Net recent average error: "
                << nn.getRecentAverageError() << std::endl;
        nn.backProp(targetVals);
		i++;
	}

	//Now test it with some non-training data
	std::vector<double> inputVals;
	inputVals = { 0.4, 0.4};
	nn.feedForward(inputVals);
	std::vector<double> resultVals;

	nn.getResults(resultVals);
		for (auto c : resultVals) {
		std::cout << "I am a neural network and 0.4 + 0.4 = " << c << std::endl;
	}
    
	return 0;
}