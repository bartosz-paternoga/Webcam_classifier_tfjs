import React from 'react';
import logo from './logo.svg';
import './App.css';


const Comp = (props) => (

  <div >


    <div className="App" onLoad = {props.main}>
      <header className="App-header" >
        <img src={logo} className="App-logo" alt="logo" /> 
        <h4 className="App-title">Webcam classifier with Tensorflow.js and REACT</h4>
      </header>     
    </div> 


    <div id="message">
       <p>To capture images for every class, press CLASS 0-2 respectively. 
       Afterwars press TRAIN, and finally PREDICT.</p>
    </div>


    <div id="webcam-wrapper" >
    <video id="webcam"  width="224" height="224" controls autoPlay ></video><br/>


    <br/><button id="1" > Class 0 </button>
    <button id="2" > Class 1 </button>
    <button id="3" > Class 2 </button><br/>

   <br/> <button  id="train" > TRAIN </button>
    <button  id="predict" > PREDICT </button>
    <button  id="reset"> RESET </button> 

  </div><br/>

</div>
);

export default Comp;
