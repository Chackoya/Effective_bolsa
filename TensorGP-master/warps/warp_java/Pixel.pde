class Pixel {
    float channel1;
    float channel2;
    float channel3;
    
    Pixel(float channel1, float channel2, float channel3) {
        this.channel1 = channel1;
        this.channel2 = channel2;
        this.channel3 = channel3;
    }
    
    public float getChannel1() {
        return channel1;
    }

    public float getChannel2() {
        return channel2;
    }

    public float getChannel3() {
        return channel3;
    }

    public void min(Pixel pixel) {
        channel1 = Math.min(channel1, pixel.channel1);
        channel2 = Math.min(channel2, pixel.channel2);
        channel3 = Math.min(channel3, pixel.channel3);
    }

    public void max(Pixel pixel) {
        channel1 = Math.max(channel1, pixel.channel1);
        channel2 = Math.max(channel2, pixel.channel2);
        channel3 = Math.max(channel3, pixel.channel3);
    }

    public void setPixel(float c1, float c2, float c3) {
      this.channel1 = c1;
      this.channel2 = c2;
      this.channel3 = c3;
    }

    public void myIf(Pixel pixel1, Pixel pixel2) {
        channel1 = (channel1 < 0) ? pixel1.channel1 : pixel2.channel1;
        channel2 = (channel2 < 0) ? pixel1.channel2 : pixel2.channel2;
        channel3 = (channel3 < 0) ? pixel1.channel3 : pixel2.channel3;
    }

    public void add(Pixel pixel) {
        channel1 += pixel.channel1;
        channel2 += pixel.channel2;
        channel3 += pixel.channel3;
    }
    
    public void mult(Pixel pixel) {
        channel1 *= pixel.channel1;
        channel2 *= pixel.channel2;
        channel3 *= pixel.channel3;
    }
    
    public void abs(Pixel pixel) {
        channel1 = Math.abs(pixel.channel1);
        channel2 = Math.abs(pixel.channel2);
        channel3 = Math.abs(pixel.channel3);
    }

    public void warp(ImageData domain, Pixel xCoord, Pixel yCoord) {
        float auxX = ImageData.imageWidth / (ImageData.maxDomain - ImageData.minDomain);
        float auxY = ImageData.imageHeight / (ImageData.maxDomain - ImageData.minDomain);

        int i1 = (int) Math.round((xCoord.getChannel1() - ImageData.minDomain) * auxX);
        int j1 = (int) Math.round((yCoord.getChannel1() - ImageData.minDomain) * auxY);
        int i2 = (int) Math.round((xCoord.getChannel2() - ImageData.minDomain) * auxX);
        int j2 = (int) Math.round((yCoord.getChannel2() - ImageData.minDomain) * auxY);
        int i3 = (int) Math.round((xCoord.getChannel3() - ImageData.minDomain) * auxX);
        int j3 = (int) Math.round((yCoord.getChannel3() - ImageData.minDomain) * auxY);

        i1 = (i1 < domain.resW) ? ((i1 >= 0) ? i1 : 0) : domain.resW - 1;
        j1 = (j1 < domain.resH) ? ((j1 >= 0) ? j1 : 0) : domain.resH - 1;
        i2 = (i2 < domain.resW) ? ((i2 >= 0) ? i2 : 0) : domain.resW - 1;
        j2 = (j2 < domain.resH) ? ((j2 >= 0) ? j2 : 0) : domain.resH - 1;
        i3 = (i3 < domain.resW) ? ((i3 >= 0) ? i3 : 0) : domain.resW - 1;
        j3 = (j3 < domain.resH) ? ((j3 >= 0) ? j3 : 0) : domain.resH - 1;

        //System.out.println("i1: "+i1+" j1: "+j1+"   i2: "+i2+" j2: "+j2+"  i3: "+i3+" j3: "+j3);

        // i*n_lines + j
        channel1 = domain.getImageRaw().get(i1 * domain.resW + j1).getChannel1();
        channel2 = domain.getImageRaw().get(i2 * domain.resW + j2).getChannel2();
        channel3 = domain.getImageRaw().get(i3 * domain.resW + j3).getChannel3();
    }

    public void mod(Pixel pixel) {
        channel1 = channel1 % pixel.channel1;
        channel2 = channel2 % pixel.channel2;
        channel3 = channel3 % pixel.channel3;
    }

}
