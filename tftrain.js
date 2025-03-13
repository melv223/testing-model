let classifier;
let data;

async function loadTrainingData() { // Obtaining training json file data 
    let response = await fetch("hand_kp.json"); // wait till file loaded
    data = await response.json();
}

async function trainModel() {
    var counter = 0;
    console.log("Loading training data...");
    // Wait for data to load before proceeding
    await loadTrainingData();
    console.log("Data loaded!");
    ml5.setBackend("webgl");
    // Set the options for the neural network
    let options = {
        task: "classification",
        debug: true,
    };
  // Initialize the neural network
    classifier = ml5.neuralNetwork(options);
    
    console.log("Processing data...")

  // Add data to the classifier
    for (let i = 0; i < data.length; i++) {
        let item = data[i];

        // Get wrist keypoint (assumed first keypoint)
        let wrist = item.keypoints[0];

        // Compute relative keypoints (differences from wrist)
        let inputs = item.keypoints.flatMap(kp => [
            kp.x, 
            kp.y, 
            kp.z
        ]);

        let output = [item.label]; // ASL letter label
        
        console.log(counter);
        counter++;
        classifier.addData(inputs, output);
    }
    console.log("Data processed!");
    console.log("Training model...");

    // Dataset is already normalized 
    //classifier.normalizeData();
    // Train the model

     const trainingOptions = {
        epochs: 70,
        batchSize: 32,
        hiddenunits: 20,
        learningRate: 0.05
    };

    await classifier.train(trainingOptions, finishedTraining);  
    classifier.save();
    
}

function finishedTraining() {
    console.log("Training complete!");
    //classify();
}

trainModel();
