import React, { Component } from 'react';
import './App.css';
import Webcam from './webcam';
import ControllerDataset from './controller_dataset.js';
import * as tf from '@tensorflow/tfjs';
import Comp from './component1';


class App extends Component {



state = {
      predd:[],
    }

  main = () => {


    let mobilenet;
    let model;
    //let predd = [];



    const NUM_CLASSES = 3;
    const webcam = new Webcam(document.getElementById('webcam'));
    const controllerDataset = new ControllerDataset(NUM_CLASSES);

    // Loads mobilenet and returns a model that returns the internal activation
    // we'll use as input to our classifier model.
    async function loadMobilenet() {
      const mobilenet = await tf.loadModel(
          'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

      // Return a model that outputs an internal activation.
      const layer = mobilenet.getLayer('conv_pw_13_relu');
      return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
    }


    // When the UI buttons are pressed, read a frame from the webcam and associate
    // it with the class label given by the button. up, down, left, right are
    // labels 0, 1, 2, 3 respectively.


    async function init() {

      await webcam.setup();
      mobilenet = await loadMobilenet();

      // Warm up the model. This uploads weights to the GPU and compiles the WebGL
      // programs so the first time we collect data from the webcam it will be
      // quick.

      tf.tidy(() => mobilenet.predict(webcam.capture()));

      const img = webcam.capture();


            // Listen for mouse events when clicking the button
            //button.addEventListener('mousedown', () => training = 10);
            //button.addEventListener('mouseup', () => training = -1);


      const button1 = document.getElementById('1');
            button1.addEventListener('mousedown', () => main0(0));
            button1.addEventListener('mouseup', () => main0(0));

      const button2 = document.getElementById('2');
            button2.addEventListener('mousedown', () => main0(1));
            button2.addEventListener('mouseup', () => main0(1));

       const button3 = document.getElementById('3');
             button3.addEventListener('mousedown', () => main0(2));
             button3.addEventListener('mouseup', () => main0(2));     


            
      function  main0 (label) {
          
      // When the UI buttons are pressed, read a frame from the webcam and associate
      // it with the class label given by the button. up, down, left, right are
      // labels 0, 1, 2, 3 respectively.

          tf.tidy(() => {
            const img = webcam.capture();
            controllerDataset.addExample(mobilenet.predict(img), label);

          });
        }

    }

    // Initialize the application.
    init();



    const button4 = document.getElementById('train');
          button4.addEventListener('click', () => train());


    async function train() {

    console.log("TRAINN");

      if (controllerDataset.xs == null) {
        alert('Add some examples before training!');
        throw new Error('Add some examples before training!');
      }

      // Creates a 2-layer fully connected model. By creating a separate model,
      // rather than adding layers to the mobilenet model, we "freeze" the weights
      // of the mobilenet model, and only train weights from the new model.
      model = tf.sequential({
        layers: [
          // Flattens the input to a vector so we can use it in a dense layer. While
          // technically a layer, this only performs a reshape (and has no training
          // parameters).
          tf.layers.flatten({inputShape: [7, 7, 256]}),
          // Layer 1
          tf.layers.dense({
            units: 100,
            activation: 'relu',
            kernelInitializer: 'varianceScaling',
            useBias: true
          }),
          // Layer 2. The number of units of the last layer should correspond
          // to the number of classes we want to predict.
          tf.layers.dense({
            units: NUM_CLASSES,
            kernelInitializer: 'varianceScaling',
            useBias: false,
            activation: 'softmax'
          })
        ]
      });

      // Creates the optimizers which drives training of the model.
      const optimizer = tf.train.adam(0.0001);
      // We use categoricalCrossentropy which is the loss function we use for
      // categorical classification which measures the error between our predicted
      // probability distribution over classes (probability that an input is of each
      // class), versus the label (100% probability in the true class)>
      model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

      // We parameterize batch size as a fraction of the entire dataset because the
      // number of examples that are collected depends on how many examples the user
      // collects. This allows us to have a flexible batch size.
      const batchSize =
          Math.floor(controllerDataset.xs.shape[0] *0.4);
      if (!(batchSize > 0)) {
        throw new Error(
            `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
      }

      // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
      model.fit(controllerDataset.xs, controllerDataset.ys, {
        batchSize,
        epochs: 20,
        callbacks: {
          onBatchEnd: async (batch, logs) => {
            console.log('Loss: ' + logs.loss.toFixed(10));


            const elem = document.getElementById("Div1");
            
            if (elem !== null){
            elem.parentNode.removeChild(elem);
          }

            const div = document.createElement('div');
            div.setAttribute("id", "Div1");
            document.body.appendChild(div);
            div.style.marginBottom = '10px';
            // Create info text


            const infoText = document.createElement('span')
            infoText.innerText = (`LOSS: ${logs.loss.toFixed(10)}`);
            div.appendChild(infoText);



            await tf.nextFrame();
          }
        }
      });



    }



    const button5 = document.getElementById('predict');
          button5.addEventListener('click', () => predict());




    //let isPredicting = false;


    async function predict() {


  // while (true) {


    const predictedClass = tf.tidy(() => {
      // Capture the frame from the webcam.
      const img = webcam.capture();

      // Make a prediction through mobilenet, getting the internal activation of
      // the mobilenet model.
      const activation = mobilenet.predict(img);

      // Make a prediction through our newly-trained model using the activation
      // from mobilenet as input.
      const predictions = model.predict(activation);

      // Returns the index with the maximum probability. This number corresponds
      // to the class the model thinks is the most probable given the input.
      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];


    console.log("classId:", classId);


      const elem = document.getElementById("Div1");
      
      if (elem !== null){
      elem.parentNode.removeChild(elem);
    }

      const div = document.createElement('div');
      div.setAttribute("id", "Div1");
      document.body.appendChild(div);
      div.style.marginBottom = '10px';
      // Create info text


      const infoText = document.createElement('span')
      infoText.innerText = (`The predicled class: ${classId}`);
      div.appendChild(infoText);



    await tf.nextFrame();
  

    //}

    }


     function reset () {
       window.location.reload();
            }  

        const button6 = document.getElementById('reset');
        button6.addEventListener('click', () => reset());



  }



  render() {


        return (
                 <Comp
                 main = {this.main}
                 reset = {this.reset}

                 />
      
       );
    }
 
}

export default App;
