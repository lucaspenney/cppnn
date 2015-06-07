#include "net.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
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

void Net::save(std::string filename) {
	//Save each neuron to a file
	rapidjson::StringBuffer s;
	rapidjson::Writer<rapidjson::StringBuffer> writer(s);
	writer.StartArray();
	for (unsigned layerNum = 0; layerNum < this->m_layers.size() - 1; ++layerNum) {
		Layer& layer = m_layers[layerNum];
		writer.StartArray();
		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			writer.StartObject();
			writer.String("weights");
			writer.StartArray();
			for (auto w : layer[n].getConnections()) {
				writer.Double(w.weight);
			}
			writer.EndArray();
			writer.String("deltaWeights");
			writer.StartArray();
			for (auto w : layer[n].getConnections()) {
				writer.Double(w.weight);
			}
			writer.EndArray();
			writer.String("gradient");
			writer.Double(layer[n].getGradient());
			writer.String("output");
			writer.Double(layer[n].getOutputVal());
			writer.EndObject();
		}
		writer.EndArray();
	}
	writer.EndArray();
	std::string data = s.GetString();

	std::ofstream file;
	file.open(filename);
	file << data;
	file.close();
}

void Net::load(std::string filename) {
	std::fstream file;
	file.open(filename);
	std::stringstream buffer;
	buffer << file.rdbuf();
	std::string data = buffer.str();

	//Parse json value and load it into network
	rapidjson::Document d;
	d.Parse(data.c_str());
	rapidjson::Value& s = d;
	for (rapidjson::SizeType i = 0; i < s.Size(); i++) {
		try {
			Layer& layer = m_layers[i];
			for (rapidjson::SizeType l = 0; l < d[i].Size(); l++) {
				std::vector<Connection>& connections = layer[l].getConnections();
				for (rapidjson::SizeType k = 0; k < d[i][l]["weights"].Size(); k++) {
					double val = d[i][l]["weights"][k].GetDouble();
					connections[k].weight = val;
				}
				for (rapidjson::SizeType k = 0; k < d[i][l]["deltaWeights"].Size(); k++) {
					double val = d[i][l]["deltaWeights"][k].GetDouble();
					connections[k].deltaWeight = val;
				}
				layer[l].setGradient(d[i][l]["gradient"].GetDouble());
				layer[l].setOutputVal(d[i][l]["output"].GetDouble());
			}
		} catch(const std::out_of_range& e) {
			std::cout << "error loading in state";
		}

	}
	m_layers.back().back().setOutputVal(1.0);
}