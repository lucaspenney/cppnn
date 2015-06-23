#include <vector>
#include <cassert>
#include <string>
#include "neuron.h"

typedef std::vector<Neuron> Layer;

class Net {
public:
	Net(const std::vector<unsigned int> &topology);
	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVals);
	void getResults(std::vector<double> &resultVals);
	double getRecentAverageError(void) const { return m_recentAverageError; }
	std::vector<Layer*> getLayers();
    void save(std::string filename);
    void load(std::string filename);
private:
	std::vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};