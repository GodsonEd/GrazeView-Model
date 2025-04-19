//Nugets needed: Microsoft.ML, Microsoft.ML.OnnxRuntime, OpenCvSharp4, OpenCvSharp4.runtime.win

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML;
using OpenCvSharp;
using static IronPython.Modules._ast;
using System.Drawing.Printing;
using SixLabors.ImageSharp.Processing;
using System.Drawing.Imaging;
using System.Drawing;
using System.Runtime.InteropServices;

namespace GrazeViewML
{
    internal class Heater
    {
        static void Mainer(string[] args)
        {
            Console.WriteLine("OpenCVSharp");

            string modelPath = "C:/Users/iruoe/Downloads/GrazeView_accur93_FinalModel.onnx"; // Replace with your ONNX model path
            //string imagePath = "C:/Users/iruoe/Downloads/184HD_001_RESIZED.png"; // Replace with your image path
            string imagePath = "C:/Users/iruoe/Downloads/QuickTest.png";
            //string imagePath = "C:/Users/iruoe/Downloads/Capture.png";

            MLContext mlContext = new MLContext();
            var sessionOptions = new SessionOptions();

            Console.WriteLine("Paths called");

            List<float[]> tilePredictions = new List<float[]>();

            using (var session = new InferenceSession(modelPath))
            {
                // 1. Get Input and Output Names (Dynamically)
                var inputName = session.InputMetadata.Keys.FirstOrDefault();
                var outputName = session.OutputMetadata.Keys.FirstOrDefault();

                Console.WriteLine("Reading Image");

                // 2. Load and Preprocess Image (OpenCVSharp4)
                Mat originalImage = Cv2.ImRead(imagePath);

                // 3. Split the image into 100x100 tiles
                List<Mat> imageTiles = SplitIntoTiles(originalImage, 100, 100, 95);
                Console.WriteLine($"Number of tiles Split: {imageTiles.Count}");

                float[] classProbabilitiesSum = new float[4];

                // 4. Process each tile and run inference
                foreach (Mat tile in imageTiles)
                {
                    // 5. Preprocess the tile and normalize
                    var inputShape = session.InputMetadata[inputName].Dimensions.ToArray();
                    int targetWidth = inputShape[2];  // Get from model
                    int targetHeight = inputShape[1]; // Get from model

                    if (tile.Width != targetWidth || tile.Height != targetHeight) //Resize only if needed
                    {
                        Cv2.Resize(tile, tile, new OpenCvSharp.Size(targetWidth, targetHeight));
                    }

                    Console.WriteLine("RGB Preprocessing");
                    float[] tileData = new float[targetWidth * targetHeight * 3]; // Assuming 3 channels (RGB or BGR)
                    for (int y = 0; y < targetHeight; y++)
                    {
                        for (int x = 0; x < targetWidth; x++)
                        {
                            Vec3b pixel = tile.At<Vec3b>(y, x);
                            tileData[y * targetWidth * 3 + x * 3 + 0] = pixel.Item1 / 255.0f; // Blue (BGR)
                            tileData[y * targetWidth * 3 + x * 3 + 1] = pixel.Item2 / 255.0f; // Green
                            tileData[y * targetWidth * 3 + x * 3 + 2] = pixel.Item0 / 255.0f; // Red
                        }
                    }

                    Console.WriteLine("Making Tensor");
                    // 6. Create Input Tensor and NamedOnnxValue
                    var inputTensor = new DenseTensor<float>(tileData, new[] { 1, targetHeight, targetWidth, 3 }); // Shape(batch_size, Height, Width, Channels)
                    var inputNamedValue = NamedOnnxValue.CreateFromTensor<float>(inputName, inputTensor);

                    Console.WriteLine("Starting Inference");
                    // 7. Run Inference
                    using (var results = session.Run(new[] { inputNamedValue }))
                    {
                        // 8. Get and process the output (same as before)
                        foreach (var outputNamedValue in results)
                        {
                            //Output tensor as a float
                            var outputTensor = outputNamedValue.AsTensor<float>();

                            // ... (Process output data for the current tile)
                            var probabilities = outputTensor.ToArray();

                            tilePredictions.Add(probabilities);
                            // Get all output tensors
                            Console.WriteLine($"Nale: {probabilities[0].ToString(".00")}");
                            Console.WriteLine($"Qufu: {probabilities[1].ToString(".00")}");
                            Console.WriteLine($"Erci: {probabilities[2].ToString(".00")}");
                            Console.WriteLine($"Bubble: {probabilities[3].ToString(".00")}");

                            // Add all four predictions to growing list
                            classProbabilitiesSum[0] += probabilities[0];
                            classProbabilitiesSum[1] += probabilities[1];
                            classProbabilitiesSum[2] += probabilities[2];
                            classProbabilitiesSum[3] += probabilities[3];
                        }

                        Console.WriteLine("Inference complete for one tile!");
                    }
                }

                // Calculate average probabilities
                float[] averageProbabilities = classProbabilitiesSum.Select(p => p / imageTiles.Count).ToArray();

                // Print the average predictions
                Console.WriteLine("\nAverage Predictions:");
                for (int i = 0; i < 4; i++)
                {
                    Console.WriteLine($"Class {i}: {averageProbabilities[i]}");
                }

                float percent_no_whitespace = averageProbabilities[0] + averageProbabilities[1] + averageProbabilities[2] + averageProbabilities[3];

                Console.WriteLine("\nNormalized Predictions:");
                Console.WriteLine($"Nale: {((averageProbabilities[0] / percent_no_whitespace) * 100).ToString(".00") + "%"}");
                Console.WriteLine($"Qufu: {((averageProbabilities[1] / percent_no_whitespace) * 100).ToString(".00") + "%"}");
                Console.WriteLine($"Erci: {((averageProbabilities[2] / percent_no_whitespace) * 100).ToString(".00") + "%"}");
                Console.WriteLine($"Bubble: {((averageProbabilities[3] / percent_no_whitespace) * 100).ToString(".00") + "%"}");

                //Creates the heatmap Mat, can edit the transparency and image size through the params
                Mat compositeHeatmapImage = CreateCompositeHeatMap(originalImage, imageTiles, tilePredictions, 100, 100, 95, 0.5f, 750, 750);

                // Display the composite heatmap
                Cv2.ImShow("Composite Heatmap", compositeHeatmapImage);
                Cv2.WaitKey(0);

                //Creates the bitmap from the Mat image and saves it
                Bitmap Heatmap = MatToBitmap(compositeHeatmapImage);
                //Heatmap.Save("C:/Users/iruoe/Downloads/compositeHeatMapImage.png", System.Drawing.Imaging.ImageFormat.Png);
                Console.WriteLine("Bitmap image saved.");

            }
            Console.ReadKey();
        }

        static List<Mat> SplitIntoTiles(Mat image, int tileWidth, int tileHeight, double whitespaceThreshold)
        {
            List<Mat> tiles = new List<Mat>();
            int totalTiles = 0;
            for (int y = 0; y < image.Height; y += tileHeight)
            {
                for (int x = 0; x < image.Width; x += tileWidth)
                {
                    totalTiles++;
                    int width = Math.Min(tileWidth, image.Width - x); //means the created tiles dont have to be 100 x 100 
                    int height = Math.Min(tileHeight, image.Height - y);//if for instance we are near edge of image
                    Rect roi = new Rect(x, y, width, height);
                    Mat tile = new Mat(image, roi);

                    // Check for excessive whitespace

                    //Create a gray version of tile
                    Mat grayTile = new Mat();
                    Cv2.CvtColor(tile, grayTile, ColorConversionCodes.BGR2GRAY);

                    //Creates an approximate mask for white pixels
                    Mat whiteMask = new Mat(grayTile.Size(), MatType.CV_8UC1, new Scalar(0));
                    Cv2.Compare(grayTile, 240, whiteMask, CmpType.GE);

                    //Calculates percent of white pixels
                    int white_pixels = Cv2.CountNonZero(whiteMask);
                    double total_pixels = grayTile.Total();
                    double whitespace_percentage = (white_pixels / total_pixels) * 100;

                    //whole numbers not percents
                    if (whitespace_percentage < whitespaceThreshold)
                    {
                        tiles.Add(tile);
                    }
                }
            }
            Console.WriteLine($"Total Tiles: {totalTiles}");

            return tiles;
        }

        static Mat CreateCompositeHeatMap(Mat originalImage, List<Mat> imageTiles, List<float[]> tilePredictions, int tileWidth, int tileHeight, double whitespaceThreshold, float transparency, int resizeWidth, int resizeHeight)
        {
            // Clone the original image to keep it intact and create a new image for the heatmap
            Mat compositeHeatmapImage = originalImage.Clone();

            List<int> validTileIndices = new List<int>(); // To track the indices of valid (non-whitespace) tiles
            List<float[]> validTilePredictions = new List<float[]>(); // To store predictions for valid tiles

            // Loop through all the tiles in the image and collect predictions for non-whitespace tiles
            int tileIndex = 0;
            int increment = 0;
            for (int y = 0; y < originalImage.Height; y += tileHeight)
            {
                for (int x = 0; x < originalImage.Width; x += tileWidth)
                {
                    int width = Math.Min(tileWidth, originalImage.Width - x);
                    int height = Math.Min(tileHeight, originalImage.Height - y);
                    Rect roi = new Rect(x, y, width, height);
                    Mat tile = new Mat(originalImage, roi);

                    // Check if the tile contains too much whitespace
                    Mat grayTile = new Mat();
                    Cv2.CvtColor(tile, grayTile, ColorConversionCodes.BGR2GRAY);

                    Mat whiteMask = new Mat(grayTile.Size(), MatType.CV_8UC1, new Scalar(0));
                    Cv2.Compare(grayTile, 240, whiteMask, CmpType.GE);

                    int white_pixels = Cv2.CountNonZero(whiteMask);
                    double total_pixels = grayTile.Total();
                    double whitespace_percentage = (white_pixels / total_pixels) * 100;

                    if (whitespace_percentage < whitespaceThreshold)
                    {
                        // If it's a valid tile, add its index to the valid list
                        Console.WriteLine(increment);
                        validTileIndices.Add(tileIndex);
                        validTilePredictions.Add(tilePredictions[increment]); // Add the corresponding prediction

                        increment++;
                    }

                    tileIndex++;
                }
            }

            // Now, let's apply the heatmap only on valid tiles
            int validTileIndex = 0;
            tileIndex = 0;

            // Loop through the image tiles again and apply heatmap color for non-whitespace tiles
            for (int y = 0; y < originalImage.Height; y += tileHeight)
            {
                for (int x = 0; x < originalImage.Width; x += tileWidth)
                {
                    int width = Math.Min(tileWidth, originalImage.Width - x);
                    int height = Math.Min(tileHeight, originalImage.Height - y);
                    Rect roi = new Rect(x, y, width, height);
                    Mat tile = new Mat(originalImage, roi);

                    // Check if this tile is in the list of valid (non-whitespace) tiles
                    if (validTileIndices.Contains(tileIndex))
                    {
                        // Get the prediction for this tile from validTilePredictions
                        float[] probabilities = validTilePredictions[validTileIndex];

                        // Find the class with the highest probability for this tile
                        int predictedClass = Array.IndexOf(probabilities, probabilities.Max());

                        // Assign a specific color to the tile based on the predicted class
                        Scalar colorTint = predictedClass switch
                        {
                            0 => new Scalar(255, 0, 0),    // Blue for Nale
                            1 => new Scalar(0, 255, 0),    // Green for Qufu
                            2 => new Scalar(0, 0, 255),    // Red for Erci
                            3 => new Scalar(0, 255, 255),  // Yellow for Bubble
                            _ => new Scalar(0, 0, 0)       // No tint for unknown classes (shouldn't happen)
                        };

                        // Apply semi-transparency (alpha blending)
                        float alpha = transparency; // increasing % makes colors brighter, add an f to ur number for the method params ex: 0.5f for 50%, 0.7f for 70%

                        // Loop through the pixels of the tile and apply the color to the composite image
                        for (int i = 0; i < height; i++)
                        {
                            for (int j = 0; j < width; j++)
                            {
                                // Get the corresponding pixel from the original image
                                Vec3b originalPixel = originalImage.At<Vec3b>(y + i, x + j);
                                Vec3b overlayPixel = new Vec3b(
                                    (byte)colorTint.Val0,  // Blue channel
                                    (byte)colorTint.Val1,  // Green channel
                                    (byte)colorTint.Val2   // Red channel
                                );

                                // Blend the original pixel with the overlay color using alpha blending
                                Vec3b blendedPixel = new Vec3b(
                                    (byte)(originalPixel.Item0 * (1 - alpha) + overlayPixel.Item0 * alpha),
                                    (byte)(originalPixel.Item1 * (1 - alpha) + overlayPixel.Item1 * alpha),
                                    (byte)(originalPixel.Item2 * (1 - alpha) + overlayPixel.Item2 * alpha)
                                );

                                // Apply the blended pixel back into the composite heatmap image
                                compositeHeatmapImage.At<Vec3b>(y + i, x + j) = blendedPixel;
                            }
                        }

                        validTileIndex++;  // Move to the next valid tile
                    }

                    tileIndex++;  // Move to the next tile index
                }
            }

            // Now, resize the composite heatmap image to 750x750
            Mat resizedHeatmap = new Mat();
            Cv2.Resize(compositeHeatmapImage, resizedHeatmap, new OpenCvSharp.Size(resizeWidth, resizeHeight));

            // Convert Mat to Bitmap
            return resizedHeatmap;
        }
        public static Bitmap MatToBitmap(Mat mat)
        {
            // Convert the Mat to a byte array
            byte[] byteArray = mat.ToBytes();

            // Create a MemoryStream from the byte array
            using (MemoryStream stream = new MemoryStream(byteArray))
            {
                // Load the byte array into a Bitmap object
                return new Bitmap(stream);
            }
        }
    }
}
