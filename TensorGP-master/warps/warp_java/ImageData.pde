public class ImageData {
    public static final String P_IMGWIDTH = "imgwidth";
    public static final String P_IMGHEIGHT = "imgheight";
    public static final int minDomain = -1;
    public static final int maxDomain = 1;
    public static final int imageWidth = 256;
    public static final int imageHeight = 256;
    
    
    /**
     *
     */
     
    public int resW, resH;
    private ArrayList<Pixel> imageRaw;

    public ImageData(int w, int h) {
        imageRaw = new ArrayList<Pixel>(w * h);
        this.resW = w;
        this.resH = h;

        for (int i = 0; i < resW * resH; i++) {
            imageRaw.add(new Pixel(0, 0, 0));
        }
    }

    public ArrayList<Pixel> getImageRaw() {
        return imageRaw;
    }
    
    
    

    public void myX() {
      for(int i = 0; i < resW; ++i) {
        float var = ((float)i / (float)resW) * 2.0 - 1.0;
        for(int j = 0; j < resH; ++j) {
           imageRaw.get(j * resW + i).setPixel(var, var, var);
        }
      }
    }
    
    public void myY() {
      for(int j = 0; j < resH; ++j) {
        float var = ((float)j / (float)resH) * 2.0 - 1.0;
        for(int i = 0; i < resW; ++i) {
          imageRaw.get(j * resW + i).setPixel(var, var, var);
        }
      }
    }
    
    public void scalar(float r) {
      for(int i = 0; i < resW * resH; ++i) {
        imageRaw.get(i).setPixel(r, r, r);
      }
    }
    
    public void scalar(float c1, float c2, float c3) {
      for(int i = 0; i < resW * resH; ++i) {
        imageRaw.get(i).setPixel(c1, c2, c3);
      }
    }
    


    public void min(ImageData anotherImage) {
        for (int i = 0; i < resW * resH; i++)
            imageRaw.get(i).min(anotherImage.getImageRaw().get(i));
    }

    public void max(ImageData anotherImage) {
        for (int i = 0; i < resW * resH; i++)
            imageRaw.get(i).max(anotherImage.getImageRaw().get(i));
    }

    public void myIf(ImageData image1, ImageData image2) {
        for (int i = 0; i < resW * resH; i++)
            imageRaw.get(i).myIf(image1.getImageRaw().get(i), image2.getImageRaw().get(i));
    }

    public void add(ImageData anotherImage) {
        for (int i = 0; i < resW * resH; i++)
            imageRaw.get(i).add(anotherImage.getImageRaw().get(i));
    }

    public void warp(ImageData anotherImage1, ImageData anotherImage2, ImageData anotherImage3) {
        for (int i = 0; i < resW * resH; i++) {

            Pixel xCoord = anotherImage1.getImageRaw().get(i);    // get x from image A
            Pixel yCoord = anotherImage2.getImageRaw().get(i);    // get y from image B


            imageRaw.get(i).warp(anotherImage3, xCoord, yCoord);
        }
    }
    
    public void mult(ImageData anotherImage) {
        for (int i = 0; i < resW * resH; i++)
            imageRaw.get(i).mult(anotherImage.getImageRaw().get(i));
    }

    public void abs() {
        for (int i = 0; i < resW * resH; i++)
            imageRaw.get(i).abs(imageRaw.get(i));
    }
    
    public int dom2col(double value) {

        value = (value > minDomain) ? value : minDomain;
        value = (value < maxDomain) ? value : maxDomain;

        return (int) ((value + maxDomain) / (2 * maxDomain) * 255);
    }
    
    public void render() {
      loadPixels();
      for (int i = 0; i < resW * resH; i++) {
        pixels[i] = color(dom2col(imageRaw.get(i).getChannel3()), dom2col(imageRaw.get(i).getChannel2()), dom2col(imageRaw.get(i).getChannel1()));
        //pixels[i] = color(255.0, 0.0, 0.0);
      }
      updatePixels();
    }
    
    
    
    
}
