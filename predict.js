let nn;
let handpose;
let video;
let predictions = [];

function setup() {
    createCanvas(400, 300);
    video = createCapture(VIDEO);
    video.hide();

    handpose = ml5.handpose(video, () => {
        console.log("HandPose Model Loaded");
    });

    handpose.on("predict", results => {
        predictions = results;
    });

    nn = ml5.neuralNetwork({ task: 'classification' });
    nn.load('model.json', () => console.log("Model Loaded!"));
}

function draw() {
    background(220);
    image(video, 0, 0, width, height);

    if (predictions.length > 0) {
        let inputs = [];
        for (let i = 0; i < 21; i++) {
            inputs.push(predictions[0].landmarks[i][0]); // x-coordinates
            inputs.push(predictions[0].landmarks[i][1]); // y-coordinates
        }
        classifyHand(inputs);
    }
}

function classifyHand(inputData) {
    nn.classify(inputData, (err, result) => {
        if (err) console.error(err);
        else console.log("Predicted Letter:", result[0].label);
    });
}
