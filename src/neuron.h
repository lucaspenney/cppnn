#include <vector>
#include <cstdlib>
#include <cmath>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned index);
	void feedForward(const Layer& prevLayer);
	double getOutputVal();
	void setOutputVal(double val);
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	void calculateOutputGradients(double targetVal);
	void calculateHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);
	std::vector<Connection> getConnections();
private:
	double sumDOW(const Layer& nextLayer);
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	unsigned int m_index;
	double m_gradient;
	static double eta; //training rate (0.0-1.0)
	static double alpha; //momentum
};