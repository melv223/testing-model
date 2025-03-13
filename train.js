
//<video id="video" width="400" height="300" autoplay></video>

//<button onclick="predictHand()">Predict</button>
let nn;
let trainingData = [];
let batchSize = 200; // Adjust batch size based on performance

function preload() {
    fetch('hand_kp.json')
        .then(response => response.json())
        .then(data => {
            console.log("Dataset Loaded. Total Samples:", data.length);
            trainingData = data;
            trainInBatches(0);
        });
}

function trainInBatches(startIndex) {
    if (startIndex >= trainingData.length) {
        console.log("Training Complete!");
        nn.normalizeData();
        nn.train({ epochs: 50 }, finishedTraining);
        return;
    }

    let endIndex = Math.min(startIndex + batchSize, trainingData.length);
    console.log(`Training Batch: ${startIndex} - ${endIndex}`);

    for (let i = startIndex; i < endIndex; i++) {
        nn.addData(trainingData[i].inputs, [trainingData[i].label]);
    }

    setTimeout(() => trainInBatches(endIndex), 100); // Prevent UI freeze
}

function setup() {
    nn = ml5.neuralNetwork({
        inputs: 42, // 21 hand keypoints * (x, y)
        outputs: 1,
        task: 'classification',
        debug: true
    });

    preload();
}

function finishedTraining() {
    console.log("Training Finished!");
    nn.save();  // Saves model.json and model.weights.bin
}
