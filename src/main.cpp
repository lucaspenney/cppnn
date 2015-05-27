#include <iostream>
#include <vector>
#include <cmath>

#include "net.h"

int main() {
	std::vector<unsigned int> topology;
	topology.push_back(2);
	topology.push_back(4);
	topology.push_back(1);
	Net nn(topology);

	int i = 0;
	while (i < 10000) {
		std::vector<double> inputVals;
		int x = rand() % 2;
		int y = rand() % 2;
		int z =  x | y;

		inputVals = {(double)x, (double)y};
		nn.feedForward(inputVals);
		std::vector<double> targetVals;
		targetVals = {(double)z};
		nn.backProp(targetVals);
		std::vector<double> resultVals;

		nn.getResults(resultVals);
		std::cout << "XOR " << x << " " << y << " should be " << z <<  std::endl;
		for (auto c : resultVals) {
			std::cout << c << ' ';
		}
		i++;
	}



    
	return 0;
}