#include <iostream>
#include <stdlib.h>   
#include <time.h> 
#include <fstream>
#include <chrono>
#include <string>

using namespace std;

/*
 * The NLayerPerceptron class implements a n-connection layer perceptron network using Sigmoid activation functions. It
 * consist of 1 input activation layer, n - 1 hidden activation layers, and 1 output activation layer, all
 * of variable activation number. The perceptron trains using steepest gradient descent and is optimized
 * using backpropagation. Configuration and outputs are done through files.
 *
 * @author Angelina Hu
 * @version 3.18.22
 */
class NLayerPerceptron
{
   #define THETA_OFFSET 1     // first element -> second layer
   #define PSI_OFFSET 1       // first element -> second layer
   #define WEIGHTS_OFFSET 2   // last index = numLayers - 2

   public: int* activationLayerNum, numLayers;   // numLayers is # of activation (!connection) layers
   public: string inFile, inWeightFile, outFile, outWeightFile, controlFile, *inFiles, inRunFile;
   public: ifstream fileReader;
   public: ofstream fileWriter;
   public: double** a, *** w, * input;

         // training only variables
   public: int Nmax, numTestCases, saveInterval;
   public: double errorThreshold, lambda;
   public: double** Psi, ** Theta;
   public: double randWeightLowerBound, randWeightUpperBound;
   public: double E, ** inputs, ** T;
   public: bool isTrainingMode, isRandomWeights, isRunningWithInfo;

   /*
    * Constructor for a NLayerPerceptron object. Configures information, echos information,
    * allocates memory, and populates weights.
    *
    * @param masterControlFile   the master control file that decides what the NLayer Perceptron executes.
    */
   public: NLayerPerceptron(string masterControlFile)
   {
      controlFile = masterControlFile;
      configureInfo();
      allocateMemory(isTrainingMode);
      populateWeights(isRandomWeights);
   } // public NLayerPerceptron()

   /*
    * Echos configuration info to outFile.
    */
   public: void echo()
   {
      // open output file
      fileWriter.open(outFile, ios::app);

      // toggles
      switch (isTrainingMode)
      {
      case true:
         fileWriter << "Training\n";
         break;
      case false:
         fileWriter << "Running\n";
         break;
      } // switch (isTrainingMode)

      switch (isRandomWeights)
      {
      case true:
         fileWriter << "Randomized\n";
         break;
      case false:
         fileWriter << "Loaded\n";
         break;
      } // switch (isRandomWeights)

      switch (isRunningWithInfo)
      {
      case true:
         fileWriter << "Info\n";
         break;
      case false:
         fileWriter << "No_Info\n";
         break;
      } // switch (isRunningWithInfo)

      // activation layer sizes
      fileWriter << "\n# Layers: " << numLayers << "\n";

      for (int alpha = 0; alpha < numLayers - 1; alpha++)
      {
         fileWriter << activationLayerNum[alpha] << " - ";
      }
      fileWriter << activationLayerNum[numLayers - 1];
      fileWriter << "\n";

      // random weight range
      if (isRandomWeights)
      {
         fileWriter << "Weight Range: " << randWeightLowerBound << " - " << randWeightUpperBound << "\n\n";
      }

      // training parameters
      if (isTrainingMode)
      {
         fileWriter << "Lambda: " << lambda << "\n# Test Cases: " << numTestCases;
         fileWriter << "\nErrorThreshold: " << errorThreshold << "\nNmax: " << Nmax << "\n\n";
      }

      // related file names
      fileWriter << "Output File: " << outFile << "\n";
      if (!isRandomWeights)
      {
         fileWriter << "Input Weight File: " << inWeightFile << "\n";
      }
      fileWriter << "Output Weight File: " << outWeightFile << "\n\n";

      // closing outFile
      fileWriter.close();
   } // public: void echo()

   /*
    * Configures information from a file.
    */
   public: void configureInfo()
   {
      string inLine;

      // read input file from control File
      fileReader.open(controlFile);
      if (fileReader.is_open())
      {
         fileReader >> inFile;
         cout << "\n" << controlFile << " opened successfully";
      }
      else
      {
         inFile = "defaultFile.txt";
         cout << "\n" << controlFile << " not opened successfully";
      }
      fileReader.close();

      // open input file
      fileReader.open(inFile);
      if (fileReader.is_open())
      {
         cout << "\n" << inFile << " opened successfully";
      }
      else
      {
         cout << "\n" << inFile << " not opened successfully";
      }

      // read configuration from file
      // toggles
      fileReader >> inLine;
      if (inLine == "Running")
      {
         isTrainingMode = false;
      }
      else if (inLine == "Training")
      {
         isTrainingMode = true;
      }

      fileReader >> inLine;
      if (inLine == "Randomized")
      {
         isRandomWeights = true;
      }
      else
      {
         isRandomWeights = false;
      }

      fileReader >> inLine;
      if (inLine == "Info")
      {
         isRunningWithInfo = true;
      }
      else
      {
         isRunningWithInfo = false;
      }

      // activation layer # and sizes
      fileReader >> inLine >> inLine >> numLayers;
      activationLayerNum = new int[numLayers];
      for (int alpha = 0; alpha < numLayers - 1; alpha++)
      {
         fileReader >> activationLayerNum[alpha] >> inLine;
      }
      fileReader >> activationLayerNum[numLayers - 1];

      // random weight range
      if (isRandomWeights)
      {
         fileReader >> inLine >> inLine >> randWeightLowerBound >> inLine >> randWeightUpperBound;
      }

      // training parameters
      if (isTrainingMode)
      {
         fileReader >> inLine >> lambda >> inLine >> inLine >> inLine >> numTestCases >> inLine >> inLine >> errorThreshold >> inLine >> Nmax;
         fileReader >> inLine >> inLine >> saveInterval;
      }

      // other file names
      fileReader >> inLine >> inLine >> outFile;
      if (!isRandomWeights)
      {
         fileReader >> inLine >> inLine >> inLine >> inWeightFile;
      }
      fileReader >> inLine >> inLine >> inLine >> outWeightFile;
      // echo();
   } // public: void configureInfo()

   /*
    * Allocates memory.
    *
    * @param:
    *    isTrainingMode   if true, memory is allocated for training.
    */
   public: void allocateMemory(bool isTrainingMode)
   {
      a = new double* [numLayers];
      w = new double** [numLayers - 1];
      input = new double[activationLayerNum[0]];

      for (int alpha = 0; alpha < numLayers; alpha++)
      {
         a[alpha] = new double[activationLayerNum[alpha]];
      }

      for (int n = 0; n < numLayers - 1; n++)
      {
         w[n] = new double* [activationLayerNum[n]];

         for (int beta = 0; beta < activationLayerNum[n]; beta++)
         {
            w[n][beta] = new double[activationLayerNum[n + 1]];
         } // for (int beta = 0; beta < activationLayerNum[n]; beta++)
      } // for (int n = 0; n < numLayers - 1; n++)

      // allocating memory for training
      if (isTrainingMode)
      {
         Theta = new double* [numLayers - THETA_OFFSET];
         Psi = new double* [numLayers - PSI_OFFSET];
         inFiles = new string[numTestCases + 1];

         // allocating memory
         for (int alpha = 0; alpha < numLayers - 1; alpha++)
         {
            Theta[alpha] = new double[activationLayerNum[alpha + THETA_OFFSET]];
            Psi[alpha] = new double[activationLayerNum[alpha + PSI_OFFSET]];
         }

         // input table
         inputs = new double* [numTestCases];
         for (int t = 0; t < numTestCases; t++)
         {
            inputs[t] = new double[activationLayerNum[0]];
         }

         // truth table
         T = new double* [numTestCases];
         for (int t = 0; t < numTestCases; t++)
         {
            T[t] = new double[activationLayerNum[numLayers - 1]];
         }
      } // if (isTrainingMode)
   } // public: void allocateMemory(bool isTrainingMode)

   /*
    * Randomizes the values of the weights for training.
    *
    * @precondition:
    *    randWeightLowerBound < randWeightUpperBound
    */
   public: void randomizeWeights()
   {
      srand(time(NULL));
      for (int n = 0; n < numLayers - 1; n++)
      {
         for (int beta = 0; beta < activationLayerNum[n]; beta++)
         {
            for (int gamma = 0; gamma < activationLayerNum[n + 1]; gamma++)
            {
               w[n][beta][gamma] = randNumber(randWeightLowerBound, randWeightUpperBound);
            }
         } // for (int beta = 0; beta < activationLayerNum[n]; beta++)
      } // for (int n = 0; n < numLayers - 1; n++)
   } // public: void randomizeWeights()

   /*
    * Saves the values of all weights in the Perceptron in order in outWeightFile.
    *
    * @param:
    *    fileName   the file to which the weights are saved
    */
   public: void saveWeights(string fileName)
   {
      // open & wipe file
      fileWriter.open(fileName, ios::trunc);

      // Layer configuration
      fileWriter << "# Layers: " << numLayers << "\n";

      for (int alpha = 0; alpha < numLayers - 1; alpha++)
      {
         fileWriter << activationLayerNum[alpha] << " - ";
      } // for (int alpha = 0; alpha < numLayers; alpha++)
      fileWriter << activationLayerNum[numLayers - 1];

      fileWriter << "\n";

      // weight table
      for (int n = 0; n < numLayers - 1; n++)
      {
         // layer header
         fileWriter << "\nWeights for n = " << n << "\n";

         // weights
         for (int beta = 0; beta < activationLayerNum[n]; beta++)
         {
            for (int gamma = 0; gamma < activationLayerNum[n + 1]; gamma++)
            {
               fileWriter << w[n][beta][gamma] << " ";
            }

            fileWriter << "\n";
         } // for (int beta = 0; beta < activationLayerNum[n]; beta++)
      } // for (int n = 0; n < numLayers - 1; n++)

      // closing fileName
      fileWriter.close();
   } // public: void saveWeights()

   /*
    * Runs or trains the perceptron depending on the toggleable training mode.
    */
   public: void execute()
   {
      if (isTrainingMode)
      {
         auto startTime = chrono::high_resolution_clock::now();
         train();
         auto done = chrono::high_resolution_clock::now();
         fileWriter.open(outFile, ios::app);
         fileWriter << "Time taken (min): " << chrono::duration_cast<std::chrono::milliseconds>(done - startTime).count()/60000.0;
         fileWriter.close();
      }
      else
      {
         auto startTime = chrono::high_resolution_clock::now();
         run(input, isRunningWithInfo);
         auto done = chrono::high_resolution_clock::now();
         fileWriter.open(outFile, ios::app);
         fileWriter << "\n\nTime taken (min): " << chrono::duration_cast<std::chrono::milliseconds>(done - startTime).count() / 60000.0;
         fileWriter.close();
      }
   } // public: void execute()

   /*
    * Populates weights by loading inputs into weights or randomizing them.
    *
    * @precondition:
    *    inFile is already open through fileReader
    *
    * @param:
    *    isRandomWeights   if true, randomizes weight values;
    *                      otherwise, loads weights from inWeightFile
    */
   public: void populateWeights(bool isRandomWeights)
   {
      // read inputs (& truth table) from file if training
      if (isTrainingMode)
      {
         // inputs & truth table for training
         for (int t = 0; t < numTestCases; t++)
         {
            fileReader >> inFiles[t];

            for (int i = 0; i < activationLayerNum[numLayers - 1]; i++)
            {
               fileReader >> T[t][i];
            } // for (int i = 0; i < outputActivationNum; i++)
         } // for (int t = 0; t < numTestCases; t++)

         // closing inFile
         fileReader.close();

         for (int t = 0; t < numTestCases; t++)
         {
            fileReader.open(inFiles[t]);

            for (int k = 0; k < activationLayerNum[0]; k++)
            {
               fileReader >> inputs[t][k];
            }

            // closing inFiles[t]  
            fileReader.close();
         } // for (int t = 0; t < numTestCases; t++)
      } // if (isTrainingMode)
      else
      {
         // reading file name for input activations
         fileReader >> inRunFile;

         // closing inFile;
         fileReader.close();

         // read input layer for running
         fileReader.open(inRunFile);

         for (int k = 0; k < activationLayerNum[0]; k++)
         {
            fileReader >> input[k];
         }

         // closing inRunFile
         fileReader.close();
      }

      if (isRandomWeights)
      {
         randomizeWeights();
      }
      else
      {
         // read weights from inWeightFile
         fileReader.open(inWeightFile);

         string inLine;

         // layer configuration header
         for (int alpha = 0; alpha <= numLayers; alpha++)
         {
            fileReader >> inLine >> inLine;
         }

         // weights tables
         for (int n = 0; n < numLayers - 1; n++)
         {
            // reading header (Ex. "Weights for n = 0")
            fileReader >> inLine >> inLine >> inLine >> inLine >> inLine;
            // weights in layer n
            for (int beta = 0; beta < activationLayerNum[n]; beta++)
            {
               for (int gamma = 0; gamma < activationLayerNum[n + 1]; gamma++)
               {
                  fileReader >> w[n][beta][gamma];
               }
            }
         } // for (int n = 0; n < numLayers - 1; n++)

         // closing inWeightFile;
         fileReader.close();
      } // if (isRandomWeights){...} else
   } // public: void populateWeights(bool isRandomWeights)

   /*
    * Calculates the activations and output for an input layer without altering
    * the original input array. NOT FOR TRAINING.
    *
    * @precondition:
    *    inputLayer.length == activationLayerNum[0]
    *
    * @param:
    *    inputLayer   an input layer to calculate activations for
    */
   public: void run(double* inputLayer)
   {
      a[0] = inputLayer;

      for (int alpha = 1; alpha < numLayers; alpha++)
      {
         for (int beta = 0; beta < activationLayerNum[alpha]; beta++)
         {
            double theta = 0.0;

            for (int gamma = 0; gamma < activationLayerNum[alpha - 1]; gamma++)
            {
               theta += a[alpha - 1][gamma] * w[alpha - 1][gamma][beta];
            }

            a[alpha][beta] = f(theta);
         } // for (int beta = 0; beta < activationLayerNum[alpha]; beta++)
      } // for (int alpha = 1; alpha < numLayers; alpha++)
   } // public: void run(double* inputLayer)

   /*
    * Calculates the activations and outputs for a given input layer without altering
    * the original input array. Also can save information about the network to files.
    * NOT FOR TRAINING.
    *
    * @precondition:
    *    inputLayer.length == inputActivationNum
    *
    * @param:
    *    inputLayer   a layer to calculate activations for
    *    withInfo     if true, saves info about network;
    *                 otherwise, false
    */
   public: void run(double* inputLayer, bool withInfo)
   {
      run(inputLayer);
      if (withInfo)
      {
         saveRunInfo(inputLayer);
      }
   } // public: void run(double* inputLayer, bool withInfo)

   /*
    * Saves information about the Perceptron including the number of activations
    * in the input and hidden layers, the values of the input activations, and
    * the calculated output to output files.
    *
    * @param:
    *    inputs   array of input activation values
    */
   public: void saveRunInfo(double inputLayer[])
   {
      // weights
      /*saveWeights(outWeightFile);*/

      // wipe contents of file if exists
      clear(outFile);

      // config info
      echo();

      // inputs & outputs to outFile
      fileWriter.open(outFile, ios::app);
      fileWriter << "\nInput:\n" << inRunFile << "\n";

      fileWriter << "\nOutput:\n";
      for (int i = 0; i < activationLayerNum[numLayers - 1]; i++)
      {
         fileWriter << a[numLayers - 1][i] << " ";
      }
      fileWriter.close();
   } // public: void saveRunInfo(double inputLayer[])

   /*
    * Saves information about the Perceptron to a file after training. Includes activation layer sizes and values, training parameters,
    * reason for terminating training, and the inputs and outputs of the training cases to output files.
    *
    * @param:
    *    inputs           array of test case input activation values
    *    T                truth table values
    *    Nmax             maximum # of iterations for training
    *    errorThreshold   the error threshold for training, training ends if total error is <= errorThreshold
    *    n                # of iterations of training run
    *    numTestCases     # of test cases in training set
    */
   public: void saveTrainInfo(double** inputs, double** T, int Nmax, double errorThreshold, int n, int numTestCases)
   {
      // weights
      saveWeights(outWeightFile);

      // wipe contents of file if exists
      clear(outFile);

      // activation layer sizes, weight randomization range, learning factor
      echo();

      // error & stopping conditions
      fileWriter.open(outFile, ios::app);
      fileWriter << "Termination Conditions:\n";
      if (E <= errorThreshold)
      {
         // error threshold met
         fileWriter << "Total error " << E << " was below the error threshold " << errorThreshold << "\n";
         fileWriter << "Maximum Iterations: " << Nmax << "\t|\t# of Iterations Run: " << n << "\n\n";
      }
      else
      {
         // maximum iterations reached
         fileWriter << "Total error " << E << " was above the error threshold " << errorThreshold << "\n";
         fileWriter << "Maximum Iterations: " << Nmax << "\t|\t# of Iterations Run: " << n;
         fileWriter << "\nMaximum Iterations reached\n\n";
         fileWriter << "\nSave Interval (ms): " << saveInterval;
      }

      // inputs & outputs for each case
      fileWriter << "\n\nTest Cases:";
      for (int t = 0; t < numTestCases; t++)
      {
         run(inputs[t]);

         fileWriter << "\nInput Test Case #" << t + 1 << "\n" << inFiles[t];

         fileWriter << "\nOutputs (F):\n";
         for (int i = 0; i < activationLayerNum[numLayers - 1]; i++)
         {
            fileWriter << a[numLayers - 1][i] << " ";
         }

         fileWriter << "\nTrue Value (T):\n";

         for (int i = 0; i < activationLayerNum[numLayers - 1]; i++)
         {
            fileWriter << T[t][i] << " ";
         }
         fileWriter << "\n";
      } // for (t = 0; t < numTestCases; t++)
      fileWriter << "\n";

      // closing outFile
      fileWriter.close();
   } // public: void saveTrainInfo(double **inputs, ...)

   /*
    * Clears a text file of content.
    *
    * @param:
    *    fileName   name of the file to be wiped
    */
   public: void clear(string fileName)
   {
      fileWriter.open(fileName, ios::trunc);
      fileWriter.close();
   } // public: void clear(string fileName)

   /*
    * Generates a random double number within the range given.
    *
    * @param:
    *    lowerBound   the lower bound of the range of random numbers
    *    upperBound   the upper bound of the range of random numbers
    *
    * @return:
    *    a random double between lowerBound and upperBound
    */
   public: double randNumber(double lowerBound, double upperBound)
   {
      double c = (double)rand() / RAND_MAX;
      return (double)(lowerBound - c * lowerBound + c * upperBound);
   } // public double randNumber(double lowerBound, double upperBound)

   /*
    * Given an input x, returns f(x) where f is the Sigmoid function.
    *
    * @param:
    *    x   the input value
    *
    * @return:
    *    f(x)
    */
   public: double f(double x)
   {
      return 1.0 / (1.0 + exp(-x));   // f(x)= 1/(1 + e^(-x))
   } // public double f(double x)

   /*
    * Given an input x, returns df(x)/dx or f'(x) where f is the Sigmoid function.
    *
    * @param:
    *    x   the input value
    *
    * @return:
    *    df(x)/dx
   */
   public: double dfdx(double x)
   {
      double fx = f(x);
      return fx * (1.0 - fx);
   } // public double dfdx(double x)

   /*
    * Calculates and returns the error between the calculated results F
    * and the true values T.
    *
    * @param:
    *    T   the truth table values
    *
    * @return:
    *    the error between T and F
    */
   public: double calculateError(double* T)
   {
      double error = 0.0;

      for (int beta = 0; beta < activationLayerNum[numLayers - 1]; beta++)
      {
         error += 0.5 * (T[beta] - a[numLayers - 1][beta]) * (T[beta] - a[numLayers - 1][beta]);
      }

      return error;
   } // public: double calculateError(double *T)

   /*
    * Trains the neural network by repeatedly adjusting the weights through
    * steepest gradient descent. It will run until Nmax iterations are reached
    * or total error is <= errorThreshold. At the end, the termination conditions
    * are saved to outFile. All parameter arrays are not changed. Optimized
    * with backpropagation.
    *
    * @precondition:
    *    numTestCases >= 1
    *    T must have dimensions numTestCases x outputActivationNum
    *    inputs must have dimensions numTestCases x inputActivationNum
    *    errorThreshold >= 0.0
    *    Nmax >= 1
    */
   public: void train()
   {
      int iterations = 0;

      E = errorThreshold + 1.0;

      auto lastTime = chrono::high_resolution_clock::now();
      auto currentTime = chrono::high_resolution_clock::now();

      while (iterations < Nmax && E > errorThreshold)
      {
         E = 0.0;
         string weightSaveFile = outWeightFile.substr(0, '.');

         // running network & adjusting weights
         for (int j = 0; j < numTestCases; j++)
         {

            // check for saving weights
            currentTime = chrono::high_resolution_clock::now();
            if (chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime).count() >= saveInterval)
            {
               string fileName = weightSaveFile + "_iteration_" + to_string(iterations) + ".txt";
               saveWeights(fileName);
               // cout << "Weights saved\n";

               lastTime = currentTime;
            }

            optimizedTrain(inputs[j], T[j]);
         }

         // aggregating error
         for (int j = 0; j < numTestCases; j++)
         {
            run(inputs[j]);

            E += calculateError(T[j]);
         }

         iterations++;
      } // while (iterations < Nmax && E > errorThreshold)

      saveTrainInfo(inputs, T, Nmax, errorThreshold, iterations, numTestCases);
   } // public: void train()

   /*
    * Runs the network using the given inputLayer as inputs then
    * adjusts the weights through steepest gradient descent which
    * is optimized by backpropagation. All parameter are not changed.
    */
   public: void optimizedTrain(double* inputLayer, double* T)
   {
      a[0] = inputLayer;

      // running network forward
      for (int alpha = 1; alpha < numLayers; alpha++)
      {
         for (int beta = 0; beta < activationLayerNum[alpha]; beta++)
         {
            Theta[alpha - THETA_OFFSET][beta] = 0.0;

            for (int gamma = 0; gamma < activationLayerNum[alpha - 1]; gamma++)
            {
               Theta[alpha - THETA_OFFSET][beta] += a[alpha - 1][gamma] * w[alpha - 1][gamma][beta];
            }

            a[alpha][beta] = f(Theta[alpha - THETA_OFFSET][beta]);
         } // for (int beta = 0; beta < activationLayerNum[alpha]; beta++)
      } // for (int alpha = 1; alpha < numLayers; alpha++)

      // calculating rightmost Psi values
      for (int beta = 0; beta < activationLayerNum[numLayers - 1]; beta++)
      {
         Psi[numLayers - 1 - PSI_OFFSET][beta] = (T[beta] - a[numLayers - 1][beta]) *
            dfdx(Theta[numLayers - 1 - THETA_OFFSET][beta]);
      }

      // calculates hidden layer Psi values and updates all but the leftmost weights
      for (int alpha = numLayers - WEIGHTS_OFFSET; alpha > 0; alpha--)
      {
         for (int beta = 0; beta < activationLayerNum[alpha]; beta++)
         {
            double Omega = 0.0;

            for (int gamma = 0; gamma < activationLayerNum[alpha + 1]; gamma++)
            {
               Omega += Psi[alpha + 1 - PSI_OFFSET][gamma] * w[alpha][beta][gamma];
               w[alpha][beta][gamma] += lambda * Psi[alpha + 1 - PSI_OFFSET][gamma] * a[alpha][beta];
            }

            Psi[alpha - PSI_OFFSET][beta] = Omega * dfdx(Theta[alpha - THETA_OFFSET][beta]);
         } // for (int beta = 0; beta < activationLayerNum[alpha]; beta++)
      } // for (int alpha = numLayers - OFFSET_WEIGHTS; alpha > 0; alpha--)

      // updating leftmost weights
      for (int beta = 0; beta < activationLayerNum[0]; beta++)
      {
         for (int gamma = 0; gamma < activationLayerNum[1]; gamma++)
         {
            w[0][beta][gamma] += lambda * Psi[1 - PSI_OFFSET][gamma] * a[0][beta];
         }
      }
   } //public: void optimizedTrain(double* inputLayer, double* T)

}; // class NLayerPerceptron

/*
 * Creates an n-layer Perceptron and can run it on input data or train it through files.
 */
int main(int argc, char* argv[])
{
   string DEFAULT_FILE_NAME = "defaultControlFile.txt";
   string controlFileName;

   if (argc > 1)
   {
      controlFileName = argv[1];
   }
   else
   {
      controlFileName = DEFAULT_FILE_NAME;
   }

   // running network
   NLayerPerceptron NLayernet = NLayerPerceptron(controlFileName);

   NLayernet.execute();
} // int main()