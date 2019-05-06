### Objective
Given a few colored and stylized texts of English alphabet generate remaining unobserved glyphs.  

### Overview
We implement 



### Dataset
The dataset consists of stylized color images with 26 English alphabets. Each image is of size 64x64. An example image is given below

### Architecture
We divide the network into two stages each specialized for a particular task. The first network, glphyNet models the
overall glyph shape and structure. The second network, ornaNet models the appearance with color and texture, enabling the transfer of fine decorative elements. Each of these networks follow a conditional GAN architecture.   
<p align='center'>
  <img src='./architecture/glyphNet.png' alt='input' width="400" height="300"/>
  <img src='./architecture/ornaNet.png' alt='input' width="400" height="300"/>
</p>
<p align='center'>
    Fig: glyphNet (Left) and ornaNet(Right)  
</p>

### Training


### Result
The following images display the outputs from glyphNet. The first row prints the conditional input, second-row prints fake image output by generator and the third-row prints ground truth. We observe that glyphNet can correctly generalize the output shape given very few glyphs of the typeface.
<p align='center'>
  <img src='./outputs/glyphNet/1.png' alt='input' width="400" height="400"/>
  <img src='./outputs/glyphNet/10.png' alt='input' width="400" height="400"/>
</p>

The next set of images display the outputs from ornaNet. The first row prints the conditional input to glyphNet generator, second-row prints conditional input to ornaNet generator, third-row prints ornaNet prediction and fourth-row prints the ground truth. ornaNet can generalize with the image gradients patterns and color. 
<p align='center'>
  <img src='./outputs/ornaNet/1.png' alt='input' width="400" height="400"/>
  <img src='./outputs/ornaNet/10.png' alt='input' width="400" height="400"/>
</p>

### Observations

### Starting Project
1. `git clone https://github.com/somil55/CV2019.git`
2. `cd CV2019`

### Installing Virtual Environment and required packages
1. `pip install virtualenv`
2. `virtualenv venv`
3. `source venv/bin/activate`
4. `pip install -r requirements.txt`

### Exiting Project
1. Run `pip freeze > requirements.txt` if you update or install additional packages and commit
2. Run `deactivate` to exit virtual environment
