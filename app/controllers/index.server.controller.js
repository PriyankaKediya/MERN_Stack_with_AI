//
//https://github.com/PacktPublishing/Hands-on-Machine-Learning-with-TensorFlow.js/tree/master/Section5_4
//
const tf = require('@tensorflow/tfjs');
// require('@tensorflow/tfjs-node');
//load iris training and testing data
const iris = require('../../iris.json');
const irisTesting = require('../../iris-testing.json');
var lossValue;


exports.trainAndPredictWithParams = function (req, res) {
    console.log('req.body' +req.body);

    const trainingData = tf.tensor2d(iris.map(item => [
        item.sepal_length, item.sepal_width, item.petal_length,
        item.petal_width
    ]))

    const outputData = tf.tensor2d(iris.map(item => [
        item.species === "setosa" ? 1 : 0,
        item.species === "virginica" ? 1 : 0,
        item.species === "versicolor" ? 1 : 0
    ]))
    //
    //tensor of features for testing data
    let sepal_length = req.body.sepal_length;
    let sepal_width = req.body.sepal_width;
    let petal_length = req.body.petal_length;
    let petal_width = req.body.petal_width;
    let epochs = req.body.epochs;
    let learning = req.body.learning;

    const testingData = tf.tensor2d([
        [parseInt(sepal_length), 
            parseInt(sepal_width), 
            parseInt(petal_length), 
            parseInt(petal_width)]
    ]);
    console.log('testingData.dataSync()  ' +testingData.dataSync())    
    // build neural network using a sequential model
    const model = tf.sequential()

    //add the first layer
    model.add(tf.layers.dense({
        inputShape: [4], // four input neurons
        activation: "sigmoid",
        units: 5, //dimension of output space (first hidden layer)
    }))

    //add the hidden layer
    model.add(tf.layers.dense({
        inputShape: [5], //dimension of hidden layer
        activation: "sigmoid",
        units: 3, //dimension of final output (setosa, virginica, versicolor)
    }))

    //add output layer
    model.add(tf.layers.dense({
        activation: "sigmoid",
        units: 3, //dimension of final output (setosa, virginica, versicolor)
    }))

    //compile the model with an MSE loss function and Adam algorithm
    model.compile({
        loss: "meanSquaredError",
        optimizer: tf.train.adam(learning),
    })
    console.log(model.summary())

    //Train the model and predict the results for testing data
    //
    // train/fit the model for the fixed number of epochs
    async function run() {
        const startTime = Date.now()
        //train the model
        await model.fit(trainingData, outputData,
            {
                epochs: epochs,
                callbacks: { //list of callbacks to be called during training
                    onEpochEnd: async (epoch, log) => {
                        lossValue = log.loss;
                        console.log(`Epoch ${epoch}: lossValue = ${log.loss}, learning rate: ${learning}, sepal length: ${sepal_length}`);
                        elapsedTime = Date.now() - startTime;
                        console.log('elapsed time: ' + elapsedTime)
                    }
                }
            }
        )

        const results = model.predict(testingData);

        results.array().then(array => {
            console.log(array[0][0])
            var resultForData1 = array[0];
            var dataToSent = { row1: resultForData1 }
            console.log(resultForData1)
            res.status(200).send(dataToSent);

        })
    }
    run()
};