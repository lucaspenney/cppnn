
#include "neuron.h"

double Neuron::eta = 0.3;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned index) {
	this->m_index = index;
	for (unsigned int c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		//Set connection value to random
		m_outputWeights.back().weight = (rand() / double(RAND_MAX));
	}
}

void Neuron::feedForward(const Layer& prevLayer) {
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].m_outputVal * prevLayer[n].m_outputWeights[this->m_index].weight;
	}

	this->m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calculateOutputGradients(double targetVal) {
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calculateHiddenGradients(const Layer& nextLayer) {
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer& prevLayer) {
	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;

		double newDeltaWeight = //Individual input, magnified by the gradient and train rate
			eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

		neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_index].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer& nextLayer) {
	double sum = 0.0;
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::setOutputVal(double val) {
	this->m_outputVal = val;
}

double Neuron::getOutputVal() {
	return this->m_outputVal;
}

double Neuron::transferFunction(double x) {
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
	return 1.0 - x * x; //Rough approximatation of derivative of hyperbolic tangent
}

std::vector<Connection> Neuron::getConnections() {
	return this->m_outputWeights;
}