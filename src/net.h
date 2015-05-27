#include <vector>
#include <cassert>
#include "neuron.h"

typedef std::vector<Neuron> Layer;

class Net {
public:
	Net(const std::vector<unsigned int> &topology);
	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVals);
	void getResults(std::vector<double> &resultVals);
private:
	std::vector<Layer> m_layers;
	double m_error;
};