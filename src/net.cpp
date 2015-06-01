#include "net.h"

#include <iostream>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

Net::Net(const std::vector<unsigned int> &topology) {
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		this->m_layers.push_back(Layer());
		unsigned int numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1]; //numOutputs is 0 if it's the last neuron

		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			this->m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			std::cout << "neuron created" << std::endl;
		}

		//Force the bias neuron's output value to 1.0, it's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const std::vector<double> &inputVals) {
	//std::cout << inputVals.size();
	//std::cout << this->m_layers[0].size()-1;
	//assert(inputVals.size() == this->m_layers[0].size()-1);
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		this->m_layers[0][i].setOutputVal(inputVals[i]);
	}

	//Forward propogate
	for (unsigned layerNum = 1; layerNum < this->m_layers.size(); ++layerNum) {
		Layer &prevLayer = this->m_layers[layerNum - 1];
		for (unsigned n = 0; n < this->m_layers[layerNum].size() - 1; ++n) {
			this->m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const std::vector<double> &targetVals) {
	//Calculate overall network error (Root mean square of output neuron errors)
	Layer&  outputLayer = this->m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1;
	m_error = sqrt(m_error);

	 m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

	//Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calculateOutputGradients(targetVals[n]);
	}
	//Create gradients on hidden layers
	for (unsigned layerNum = this->m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer& hiddenLayer = m_layers[layerNum];
		Layer& nextLayer = m_layers[layerNum + 1];
		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}
	}

	//Update connection weights
	for (unsigned layerNum = this->m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}

}

void Net::getResults(std::vector<double> &resultVals) {
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::save() {
	//Save each neuron to a file
	rapidjson::StringBuffer s;
	rapidjson::Writer<rapidjson::StringBuffer> writer(s);
	writer.StartArray();
	for (unsigned layerNum = this->m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer& layer = m_layers[layerNum];
		writer.StartArray();
		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			writer.StartObject();
			writer.StartArray();
			for (auto w : layer[n].getConnections()) {
				writer.Double(w.weight);
			}
			writer.EndArray();
			writer.StartArray();
			for (auto w : layer[n].getConnections()) {
				writer.Double(w.weight);
			}
			writer.EndArray();
			writer.EndObject();
		}
		writer.EndArray();
	}
	writer.EndArray();
}