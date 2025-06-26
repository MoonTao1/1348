# Driving Fixation Prediction via Unsupervised Domain Adaptation for Sunny-to-Rainy Traffic Scenes

​	[note] We will release our complete code after the paper is **accepted** ✔️! Please look forward to it.🕓

## 📰 News

**[2025.2.15]** 🎈We have completed the training of the model and verified that it successfully enables the model to adapt to the rainy scenario. 

**[2025.4.07]** 🎈We have conducted more detailed experiments, performing both qualitative and quantitative comparisons with other advanced methods, as well as a parameter comparison.

**[2025.5.26]** 🎈We will submit the article to ***PRCV*** (The 8th Chinese Conference on Pattern Recognition and Computer Vision).😃

## ✨Model

<div align="center">
<img src="pic\model3.png" width=1000" height="auto" />
</div>


>The architecture of the proposed model. The model consists of three structures, two feature extractors with shared parameters, a predictor and a domain adaptation module. The target domain data does not pass through the predictor, and only the source domain data passes through the predictor to calculate the loss value. The data of the two domains need to go through the domain classifier for domain discrimination, and a classification loss is obtained.

## 💻Dataset

<div align="center">
<img src="pic\12.png" width="700" height="auto" />
</div>

<div align="center">
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Train (video/frame)</th>
      <th>Valid (video/frame)</th>
      <th>Test (video/frame)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TrafficGaze</td>
      <td>49080</td>
      <td>6655</td>
      <td>19135</td>
    </tr>
    <tr>
      <td>DrFixD-rainy</td>
      <td>52291</td>
      <td>9816</td>
      <td>19154</td>
    </tr>
  </tbody>
</table>
</div>


>The datasets are organized as follows.
<div align="center">
<table>
<tr>
    <th>TrafficGaze</th>
    <th>DrFixD-rainy</th>
  </tr>
  <tr>
    <td>
      ./TrafficGaze<br>
      &emsp;&emsp;|——fixdata<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata1.mat<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata2.mat<br>
      &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata16.mat<br>
      &emsp;&emsp;|——trafficframe<br>
      &emsp;&emsp;|&emsp;&emsp;|——01<br>
      &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|——000001.jpg<br>
      &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——02<br>
      &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——16<br>
      &emsp;&emsp;|——test.json<br>
      &emsp;&emsp;|——train.json<br>
      &emsp;&emsp;|——valid.json
    </td>
    <td>
      ./DrFixD-rainy<br>
      &emsp;&emsp;|——fixdata<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata1.mat<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata2.mat<br>
      &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——fixdata16.mat<br>
      &emsp;&emsp;|——trafficframe<br>
      &emsp;&emsp;|&emsp;&emsp;|——01<br>
      &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|——000001.jpg<br>
      &emsp;&emsp;|&emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——02<br>
      &emsp;&emsp;|&emsp;&emsp;|—— ... ...<br>
      &emsp;&emsp;|&emsp;&emsp;|——16<br>
      &emsp;&emsp;|——test.json<br>
      &emsp;&emsp;|——train.json<br>
      &emsp;&emsp;|——valid.json
    </td>
  </tr>
</table>
</div>


## 🚀 Quantitative Analysis
>COMPARISON WITH OTHER METHODS FROM TraffiicGaze TO DRFIXD(RAINY)

<div align="center">
  <table border="1" style="margin: 0 auto;">
    <thead>
      <tr>
        <th>Model</th>
        <th>AUC_B↑</th>
        <th>AUC_J↑</th>
        <th>NSS↑</th>
        <th>CC↑</th>
        <th>SIM↑</th>
        <th>KLD↓</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Itti</td>
        <td>0.8426</td>
        <td>0.8570</td>
        <td>1.5569</td>
        <td>0.3695</td>
        <td>0.2702</td>
        <td>1.6868</td>
      </tr>
      <tr>
        <td>ImageSig</td>
        <td>0.6368</td>
        <td>0.6912</td>
        <td>0.5298</td>
        <td>0.1404</td>
        <td>0.1781</td>
        <td>2.4159</td>
      </tr>
      <tr>
        <td>HFT</td>
        <td>0.7670</td>
        <td>0.7867</td>
        <td>1.0077</td>
        <td>0.2311</td>
        <td>0.2240</td>
        <td>2.0079</td>
      </tr>
      <tr>
        <td>DDC</td>
        <td>0.7359</td>
        <td>0.7608</td>
        <td>0.9433</td>
        <td>0.2492</td>
        <td>0.2454</td>
        <td>1.8950</td>
      </tr>
      <tr>
        <td>Deep Coral</td>
        <td>0.7132</td>
        <td>0.7680</td>
        <td>1.1630</td>
        <td>0.3135</td>
        <td>0.2909</td>
        <td>1.8543</td>
      </tr>
      <tr>
        <td>DRCN</td>
        <td>0.6021</td>
        <td>0.7676</td>
        <td>0.8007</td>
        <td>0.2307</td>
        <td>0.2688</td>
        <td>2.5432</td>
      </tr>
      <tr>
        <td>AT</td>
        <td>0.7498</td>
        <td>0.7440</td>
        <td>0.9608</td>
        <td>0.2578</td>
        <td>0.2634</td>
        <td>1.7583</td>
      </tr>
      <tr>
        <td>CPFE</td>
        <td>0.8147</td>
        <td>0.9201</td>
        <td>2.5645</td>
        <td>0.5720</td>
        <td>0.4722</td>
        <td>1.0185</td>
      </tr>
      <tr>
        <td>FBL</td>
        <td>0.8172</td>
        <td><strong>0.9395</strong></td>
        <td><strong>3.7492</strong></td>
        <td>0.7279</td>
        <td>0.5716</td>
        <td>0.7778</td>
      </tr>
      <tr>
        <td>Ours</td>
        <td><strong>0.8612</strong></td>
        <td>0.9320</td>
        <td>3.3928</td>
        <td><strong>0.7442</strong></td>
        <td><strong>0.5965</strong></td>
        <td><strong>0.7510</strong></td>
      </tr>
    </tbody>
  </table>
</div>







>COMPARISION OF FPS AND PARAMETERS

<div align="center">
  <table border="1" style="margin: 0 auto;">
    <thead>
      <tr>
        <th>Model</th>
        <th>Parameter</th>
        <th>FPS</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>DDC</td>
        <td><strong>9.90M</strong></td>
        <td>82.91</td>
      </tr>
      <tr>
        <td>Deep Coral</td>
        <td>378.17M</td>
        <td>35.85</td>
      </tr>
      <tr>
        <td>DRCN</td>
        <td>191.70M</td>
        <td>95.14</td>
      </tr>
      <tr>
        <td>AT</td>
        <td>11.20M</td>
        <td>170.78</td>
      </tr>
      <tr>
        <td>CPFE</td>
        <td>16.38M</td>
        <td>46.76</td>
      </tr>
      <tr>
        <td>FBL</td>
        <td>87.48M</td>
        <td>33.19</td>
      </tr>
      <tr>
        <td>Ours</td>
        <td><strong>9.90M</strong></td>
        <td><strong>173.79</strong></td>
      </tr>
    </tbody>
  </table>
</div>



## 🚀Visualisation of intermediate results
>Qualitative evaluation comparison of proposed model and the other methods from sunny dataset TrafficGaze to rainy dataset DrFixD(rainy). The circles highlight objects/areas in the driving scene that disrupt the driver's attention.

<div align="center">
<img src="pic\AM.png" width="1200" height="auto" />
</div>



>Visualization results of rainy weather forecast under complex scenarios, yellow circles indicate unrelated factors that may capture the driver's attention, including (a) The wiper blocks the line of sight, (b) Reflection from stagnant water, (c) Glare in the taillights of vehicles ahead in dim light.

<div align="center">
<img src="pic\11.png" width="600" height="auto" />
</div>



## 🛠️ Deployment **[🔁](#🔥Update)**

### 	Run train 

​	👉*If you wish to train with our model, please use the proceeding steps below.*

1. Train our model.  You can use `--category` to switch datasets, which include `TrafficGaze`, `DrFixD-rainy`, --b`  sets batch size, `--g  sets id of cuda.

```python
python main.py --network xxx --b 32 --g 0 --category xxx --root xxx
```


## 📄Cite

Welcome to our work ! 

