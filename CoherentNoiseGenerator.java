package App;

// Provided by google AI
// Not verified as correct but it does seem to make noise


// To add Fractal Brownian Motion (fBm) noise to an image in Java, you must first have a source for coherent noise, such as a Perlin or Simplex noise implementation.
// fBm is created by combining multiple layers (called octaves) of this coherent noise, each with a higher frequency and lower amplitude than the last. 
// A basic, self-contained implementation involves these steps:
// Obtain an image to modify, typically a BufferedImage object.
// Select or implement a coherent noise generator, such as Perlin or Simplex noise. For this example, we will implement a simplified version.
// Iterate through each pixel of the image.
// For each pixel, calculate the final noise value by summing up multiple noise "octaves".
// Offset the color values of each pixel by the final noise value. 
// Implementation example

// Explanation of parameters
// Octaves: The number of coherent noise layers to combine. More octaves lead to a more detailed, "crinkly" appearance. A good starting point is 4 to 8.
// Persistence: The factor by which amplitude decreases for each successive octave. A value of 0.5 is common, meaning each octave has half the amplitude of the one before it.
// Scale: Controls the overall size and "tightness" of the noise. A smaller value (like 0.01) creates broader, smoother noise patterns, while a larger value creates tighter, more erratic noise. 


// Step 1: CoherentNoiseGenerator.java
// This class provides the base noise function. A simple Perlin-style noise generator can be implemented with a gradient grid and interpolation.
// A faster, more direct approach is to use a "Value Noise" algorithm, which is essentially fBm using white noise. 
// java

import java.util.Random;

public class CoherentNoiseGenerator {

    private final int[] permutation;

    public CoherentNoiseGenerator() {
        Random rand = new Random();
        permutation = new int[512];
        int[] p = new int[256];
        for (int i = 0; i < 256; i++) {
            p[i] = i;
        }

        // Shuffle the permutation array
        for (int i = 0; i < 255; i++) {
            int j = rand.nextInt(256);
            int temp = p[i];
            p[i] = p[j];
            p[j] = temp;
        }

        // Extend permutation array to avoid modulo
        for (int i = 0; i < 256; i++) {
            permutation[i] = p[i];
            permutation[i + 256] = p[i];
        }
    }

    // A simplified 2D coherent noise function
    public double noise(double x, double y) {
        int X = (int)Math.floor(x) & 255;
        int Y = (int)Math.floor(y) & 255;

        x -= Math.floor(x);
        y -= Math.floor(y);

        double u = fade(x);
        double v = fade(y);

        int A = permutation[X] + Y;
        int B = permutation[X + 1] + Y;

        return lerp(v, lerp(u, grad(permutation[A], x, y), grad(permutation[B], x - 1, y)),
                       lerp(u, grad(permutation[A + 1], x, y - 1), grad(permutation[B + 1], x - 1, y - 1)));
    }

    private double fade(double t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    private double lerp(double t, double a, double b) {
        return a + t * (b - a);
    }

    private double grad(int hash, double x, double y) {
        int h = hash & 15;
        double u = h < 8 ? x : y;
        double v = h < 4 ? y : h == 12 || h == 14 ? x : 0;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
}