Math.tanh = Math.tanh || function(x) {
	if (x === Infinity) {
		return 1;
	} else if (x === -Infinity) {
		return -1;
	} else {
		return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
	}
}

var r = [];
for (var i = 0; i < 10000; i++) {
	var i1 = Math.random() * 0.4;
	var i2 = Math.random() * 0.4;
	var o = i1 + i2;
	r.push({
		inputs: [i1, i2],
		outputs: [o]
	});
}

//Write data to file to be used by network
var fs = require('fs');
fs.writeFile("training.json", JSON.stringify(r), function(err) {
	if (err) {
		return console.log(err);
	}

	console.log("The file was saved!");
});