
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include <optionparser.h>

#include "net.h"

double frand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

enum  optionIndex { UNKNOWN, HELP, TRAIN, LOAD };
const option::Descriptor usage[] =
{
	{UNKNOWN, 0, "", "",option::Arg::None, "USAGE: example [options]\n\n Options:" },
	{HELP, 0,"", "help",option::Arg::None, "  --help  \tPrint usage and exit." },
	{TRAIN, 0,"t","train",option::Arg::Optional, "  --train, -t  \tSpecify training data file to train network with." },
	{LOAD, 0,"l","load",option::Arg::Optional, "  --load, -l  \tSpecify network data file to load." },
	{0,0,0,0,0}
};

int main(int argc, char* argv[]) {
	//Parse cli options
	std::cout << &(*argv[1]);
	argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
	option::Stats  stats(usage, argc, argv);
	option::Option* options = new option::Option[stats.options_max];
	option::Option* buffer  = new option::Option[stats.buffer_max];
	option::Parser parse(usage, argc, argv, options, buffer);

	if (parse.error())
		return 1;

	if (options[HELP]) {
		option::printUsage(std::cout, usage);
		return 1;
	}

	std::string trainingFile = "";
	std::string stateFile = "state.json";

	for (int i = 0; i < parse.optionsCount(); ++i)
	{
		option::Option& opt = buffer[i];
		switch (opt.index())
		{
			case TRAIN:
			if (opt.arg)
				std::cout << "Training network from " << opt.arg << std::endl;
				trainingFile = opt.arg;
			break;
			case LOAD:
			if (opt.arg) 
				std::cout << "Loading network state from " << opt.arg << std::endl;
				stateFile = opt.arg;
			break;
		}
	}

	srand(time(NULL));
	std::vector<unsigned int> topology;

	topology.push_back(2);
	topology.push_back(10);
	topology.push_back(5);
	topology.push_back(1);
	Net nn(topology);

	if (stateFile.length() > 1) {
		nn.load(stateFile);	
	}

	if (trainingFile.length() > 1) {
		nn.trainFromFile(trainingFile);
	}

	if (stateFile.length() > 1) {
		nn.save(stateFile);
	}

	//Now test it with some non-training data
	std::vector<double> inputVals;
	inputVals = { 1, 0};
	nn.feedForward(inputVals);
	std::vector<double> resultVals;

	nn.getResults(resultVals);
	for (auto c : resultVals) {
		std::cout << "Inputs: 1 | 0" << std::endl;
		std::cout << "Outputs: " << c << std::endl;
		std::cout << "Average Error: " << nn.getRecentAverageError() << std::endl;
	}

	return 0;
}