void setup() {
  size(256, 256);
  noLoop();
}

void draw() {
  ImageData tensorX = new ImageData(256, 256); // x tensor
  tensorX.myX();
  
  ImageData tensorY = new ImageData(256, 256); // y tensor
  tensorY.myY();
  
  ImageData abs1 = new ImageData(256, 256); // image tensor
  abs1.myX();
  abs1.abs();
  
  ImageData scalar1 = new ImageData(256, 256);
  scalar1.scalar(0.46855265, 0.46855265, 0.46855265);
  scalar1.add(tensorY);
  //scalar1.render();
  
  ImageData scalar2 = new ImageData(256, 256);
  scalar2.scalar(0.9893352, 0.70297724, 0.44974214);
  scalar2.min(tensorX);
  //scalar2.render();
  
  scalar1.mult(scalar2);
  //scalar1.render();
  
  ImageData warpTensor = new ImageData(256, 256);
  warpTensor.warp(tensorX, scalar1, abs1);
  
  
  //expr = 'warp(abs(x), x, mult(add(scalar(0.46855265, 0.46855265, 0.46855265), y), min(scalar(0.9893352, 0.70297724, 0.44974214), x)))'
  warpTensor.render();
  
}
